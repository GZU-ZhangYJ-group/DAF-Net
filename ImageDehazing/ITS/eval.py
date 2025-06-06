import os
import torch
from torchvision.transforms import functional as F
from utils import Adder
from data import test_dataloader, inf_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time
from pytorch_msssim import ssim
import torch.nn.functional as f
import pandas as pd
from tqdm import tqdm

def _eval(model, args):
    state_dict = torch.load(args.test_model)
    #print(state_dict.keys())
    model.load_state_dict(state_dict['model'], strict=True)  # params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.mode == 'test':
        dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    else:
        dataloader = inf_dataloader(args.data_dir, batch_size=1, num_workers=0)
    # dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    # dataloader = inf_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    adder = Adder()
    model.eval()
    factor = 8
    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()
        niqe_adder = Adder()
        pi_adder = Adder()
        results = []
        for batch_idx, data in enumerate(tqdm(dataloader, desc="Processing Batches")):
        # for iter_idx, data in enumerate(dataloader):
            if args.mode == 'test':
                dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
                input_img, label_img, name = data
            else:
                dataloader = inf_dataloader(args.data_dir, batch_size=1, num_workers=0)
                input_img, name = data
                label_img = None

            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            tm = time.time()

            pred = model(input_img)[2]
            pred = pred[:, :, :h, :w]

            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            if label_img is not None:
                pred_numpy = pred_clip.squeeze(0).cpu().numpy()
                label_numpy = label_img.squeeze(0).cpu().numpy()

                label_img = (label_img).cuda()
                psnr_val = 10 * torch.log10(1 / f.mse_loss(pred_clip, label_img))

                down_ratio = max(1, round(min(H, W) / 256))
                ssim_val = ssim(f.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))),
                                f.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))),
                                data_range=1, size_average=False)
                print('%d iter PSNR_dehazing: %.2f ssim: %f' % (iter_idx + 1, psnr_val, ssim_val))
                ssim_adder(ssim_val)

                if args.save_image:
                    save_name = os.path.join(args.result_dir, name[0])
                    pred_clip += 0.5 / 255
                    pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                    pred.save(save_name)

                psnr_mimo = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
                psnr_adder(psnr_val)

                print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr_mimo, elapsed))
                # Append results for CSV
                results.append({
                    'Image Name': name[0],
                    'PSNR': psnr_val.item(),
                    'SSIM': ssim_val.item()
                })
                results_df = pd.DataFrame(results)
                results_df.to_csv(os.path.join('psnr_ssim_results.csv'), index=False)
            else:
                pred_numpy = pred_clip.squeeze(0).cpu().numpy()
                # niqe_val = 
                # niqe_adder(niqe_val)
                if args.save_image:
                    save_name = os.path.join(args.result_dir, name[0])
                    pred_clip += 0.5 / 255
                    pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                    pred.save(save_name)


        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print('The average SSIM is %.5f dB' % (ssim_adder.average()))
        print('The average NIQE is %.2f dB' % (niqe_adder.average()))
        print('The average PI is %.5f dB' % (pi_adder.average()))

        print("Average time: %f" % adder.average())

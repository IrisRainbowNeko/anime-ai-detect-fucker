import torch
import os
from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from torchvision.utils import save_image
import torch.nn.functional as F
from torchvision import transforms

from attacker import *
from torch.nn import CrossEntropyLoss

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_args():
    parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')

    parser.add_argument('inputs', type=str)
    parser.add_argument('--target', type=str, default='auto', help='[auto, ai, human, same]')
    parser.add_argument('--eps', type=float, default=8/8, help='Noise intensity ')
    parser.add_argument('--step_size', type=float, default=1.087313/8, help='Attack step size')
    parser.add_argument('--steps', type=int, default=20, help='Attack step count')

    parser.add_argument('--test_atk', action='store_true')

    return parser.parse_args()

class Attacker:
    def __init__(self, args, save_callback=None):
        self.args=args
        self.save_callback=save_callback

        print('正在加载模型...')
        self.feature_extractor = BeitFeatureExtractor.from_pretrained('saltacc/anime-ai-detect')
        self.model = BeitForImageClassification.from_pretrained('saltacc/anime-ai-detect').to(device)
        print('加载完毕')

        if args.target=='ai': #攻击成被识别为AI
            self.target = torch.tensor([1]).to(device)
        elif args.target=='human':
            self.target = torch.tensor([0]).to(device)
        else:
            self.target = torch.tensor([0]).to(device)

        dataset_mean_t = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(device)
        dataset_std_t = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(device)
        self.pgd = PGD(self.model, img_transform=(lambda x: (x - dataset_mean_t) / dataset_std_t, lambda x: x * dataset_std_t + dataset_mean_t))
        self.pgd.set_para(eps=(args.eps * 2) / 255, alpha=lambda: (args.step_size * 2) / 255, iters=args.steps)

        def loss_same(a, b):
            return -torch.exp((a[0, 0] - a[0, 1]) ** 2)
        self.pgd.set_loss(loss_same if args.target=='same' else CrossEntropyLoss())

    def save_image(self, image, noise, img_name, output_dir):
        # 缩放图片只缩放噪声
        W, H = image.size
        noise = F.interpolate(noise, size=(H, W), mode='bicubic')
        img_save = transforms.ToTensor()(image) + noise
        if self.save_callback is not None:
            self.save_callback(img_save)
        save_image(img_save, os.path.join(output_dir, f'{img_name[:img_name.rfind(".")]}_atk.png'))

    def attack_(self, image, output_dir):
        inputs = self.feature_extractor(images=image, return_tensors="pt")['pixel_values'].to(device)

        if self.args.target == 'auto':
            with torch.no_grad():
                outputs = self.model(inputs)
                logits = outputs.logits
                cls = logits.argmax(-1).item()
                target = torch.tensor([cls]).to(device)
        else:
            target = self.target

        if self.args.test_atk:
            self.test_image(inputs, 'before attack')

        atk_img = self.pgd.attack(inputs, target)

        noise = self.pgd.img_transform[1](atk_img).detach().cpu() - self.pgd.img_transform[1](inputs).detach().cpu()

        if self.args.test_atk:
            self.test_image(atk_img, 'after attack')

        return atk_img, noise

    def attack_one(self, path):
        image = Image.open(path).convert('RGB')
        atk_img, noise = self.attack_(image, os.path.dirname(path))
        self.save_image(image, noise, os.path.basename(path), os.path.dirname(path))

    def attack(self, path):
        count=0
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if (file.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))):
                        img_path = os.path.join(root, file)
                        self.attack_one(img_path)
                        count += 1
        else:
            if (path.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))):
                self.attack_one(path)
                count += 1
        print(f'总共攻击{count}张图像')

    @torch.no_grad()
    def test_image(self, img, pre_fix):
        outputs = self.model(img)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        print(pre_fix, "class:", self.model.config.id2label[predicted_class_idx], 'logits:', logits)

if __name__ == '__main__':
    args=make_args()
    attacker = Attacker(args)
    attacker.attack(args.inputs)

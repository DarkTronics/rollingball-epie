import sys
import time
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import cupyx.scipy.fft as cufft
import scipy.fft
from scipy.fft import ifft2, fft2, ifftshift, fftshift
import torch
import os
from PIL import Image
scipy.fft.set_global_backend(cufft)
import time

os.makedirs('images', exist_ok=True)
os.makedirs('pt', exist_ok=True)

lam = 632.8 * (10**(-9))
pixelnum = 1600
pixelsize = 7.4* (10**(-6))
stepnumber=5
d = 142 * (10**(-3))
k = 2 * np.pi / lam
z1 = 27 * (10**(-3))
z2 = 10 * (10**(-3))
ratio=68

xx = cp.linspace(-pixelsize*pixelnum/2, pixelsize*pixelnum/2, pixelnum)
yy = cp.linspace(-pixelsize*pixelnum/2, pixelsize*pixelnum/2, pixelnum)
[XX, YY] = cp.meshgrid(xx, yy)
fx = cp.linspace(-1/pixelsize/2, 1/pixelsize/2, pixelnum)
fy = cp.linspace(-1/pixelsize/2, 1/pixelsize/2, pixelnum)
[Fx, Fy] = cp.meshgrid(fx, fy)

hole = (cp.sqrt(XX**2+YY**2)<(1.2*10**(-3))).astype(int)

H1 = cp.exp(1j*k*d*cp.sqrt(1-(lam*Fx)**2-(lam*Fy)**2+0j))
H2 = cp.exp((-1j*k*d*cp.sqrt(1-(lam*Fx)**2-(lam*Fy)**2+0j)).real)
H3 = cp.exp(1j*k*z2*cp.sqrt(1-(lam*Fx)**2-(lam*Fy)**2+0j))

num_iteration = 100
progress_bar = True
Tx = cp.loadtxt('x.txt', unpack = True).T
Ty = cp.loadtxt('y.txt', unpack = True).T

hole = fftshift(ifft2(ifftshift(H3 * (fftshift(fft2(ifftshift(hole)))))))
r = cp.sqrt(z1**2 + XX**2 + YY**2)
Afield = hole * cp.exp(1j*k*r) * 0.1
Assumption1 = cp.ones((6000,6000)).astype(cp.complex128)

# Rolling ball parameters to scan every diffraction pattern
start_time = time.time()
count = 0
for ry in range(3):  # vertical steps (rows)
    for rx in range(6):  # horizontal steps (columns)

        for i in range(num_iteration):
            for n in range(5):
                for m in range(stepnumber):
                    # Progress output
                    if(progress_bar == True):
                        sys.stdout.write("\rIteration:"+ str(i+1)+ "  Progress:"+ str((5*n+m+1)/0.25) + "%")
                        sys.stdout.flush()

                    # Load diffraction image
                    filename = str((n + ry) * 10 + m + 1 + rx) + ".PNG"
                    print("File number:", filename)
                    diffraction_image = plt.imread(filename)
                    I = 255 * diffraction_image[1031 - 800:1031 + 800, 1065 - 800:1065 + 800]
                    I = cp.array(I)

                    # Define reconstruction patch boundaries
                    x1 = 1601 - np.round((Tx[(n + ry) * 10 + m + rx] - Tx[(n + ry) * 10 + rx]) / ratio)
                    x2 = x1 + pixelnum
                    y1 = 2601 - np.round((Ty[n + ry] - Ty[ry]) / ratio)
                    y2 = y1 + pixelnum

                    # Reconstruction patch
                    Tem2 = Assumption1[x1:x2, y1:y2]

                    # Ptychographic update
                    assump2_step1 = ifftshift(H1 * fftshift(fft2(ifftshift(Afield * Tem2))))
                    Assumption2 = fftshift(ifft2(assump2_step1))
                    Renew = cp.sqrt(abs(I)) * cp.exp(1j * cp.angle(Assumption2))
                    Reconstruction = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Renew))) * cp.conj(H1))))
                    change = Reconstruction - Afield * Tem2
                    Tem2 = Tem2 + cp.conj(Afield) / (abs(Afield)**2 + 1) * abs(Afield) / abs(Afield).max() * change
                    temp = Tem2
                    Afield = Afield + cp.conj(temp) / (abs(temp)**2 + 0.1) * abs(temp) / abs(temp).max() * change

                    # Update assumption
                    Assumption1[x1:x2, y1:y2] = Tem2
            
            fig, (ax1,ax2,ax3) = plt.subplots(1,3)
            plt.gcf().set_size_inches(13, 10)
            plot_1 = ax1.imshow(abs(Assumption1[2000:2600,2900:3500]).get(), cmap='gray')
            plot_2 = ax2.imshow(cp.angle(Assumption1[2000:2600,2900:3500]).get(), cmap='gray')
            plot_3 = ax3.imshow(abs(Afield).get(), cmap='gray')
            # plt.show(block=False)
            plt.close(fig)

        end_time = time.time()

        # --- Plot images ---
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plt.gcf().set_size_inches(13, 10)
        ax1.imshow(abs(Assumption1[2000:2600, 2900:3500]).get(), cmap='gray')
        ax2.imshow(cp.angle(Assumption1[2000:2600, 2900:3500]).get(), cmap='gray')
        ax3.imshow(abs(Afield).get(), cmap='gray')
        # plt.show(block=False)
        plt.close(fig)

        # --- Save PNGs ---
        amp_crop = abs(Assumption1[2150:2450, 3150:3450]).get()
        phase_crop = cp.angle(Assumption1[2150:2450, 3150:3450]).get()


        plt.imsave(f'images/absolute_{rx}{ry}.png', amp_crop, cmap='gray')
        plt.imsave(f'images/phase_{rx}{ry}.png', phase_crop, cmap='gray')
        plt.imsave(f'images/probe_{rx}{ry}.png', abs(Afield.get()), cmap='gray')

        amp_crop = Image.open(f'images/absolute_{rx}{ry}.png').convert('L')
        amp_crop = np.array(amp_crop).astype(np.float32) / 255.0  # Normalize to [0, 1]
        phase_crop = Image.open(f'images/phase_{rx}{ry}.png').convert('L')
        phase_crop = np.array(phase_crop).astype(np.float32) / 255.0  # Normalize to [0, 1]

        label_tensor = torch.tensor(np.stack((amp_crop, phase_crop), axis=0), dtype=torch.float32)  # shape: [2, 300, 300]

        # --- Save .pt file ---
        input_images = []
        for n in range(5):
            for m in range(stepnumber):
                filename = str((n + ry) * 10 + m + 1 + rx) + ".PNG"
                print(f"Reading diffraction pattern: {filename}")
                diffraction_image = Image.open(filename).convert('L')
                img_array = np.array(diffraction_image).astype(np.float32) / 255.0  # Normalize to [0, 1]
                input_images.append(torch.tensor(img_array, dtype=torch.float32))  # save raw diffraction pattern

                
        input_tensor = torch.tensor(np.stack(input_images, axis=0), dtype=torch.float32)  # shape: [25, h, w]

        torch.save({'input': input_tensor, 'label': label_tensor}, f'pt/data_{count}.pt')
        count += 1

total_end_time = time.time()
total_duration = total_end_time - start_time
print(f"\nTotal execution time: {total_duration:.2f} seconds")

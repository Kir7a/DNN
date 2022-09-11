from turtle import forward
import numpy as np
import torch

class DiffractiveLayer(torch.nn.Module):

    def __init__(self, λ = 532e-9, N_pixels = 400, pixel_size = 20e-6, dz = 0.01):
        super(DiffractiveLayer, self).__init__()

        fx = np.fft.fftshift(np.fft.fftfreq(N_pixels, d = pixel_size))
        fy = np.fft.fftshift(np.fft.fftfreq(N_pixels, d = pixel_size))
        fxx, fyy = np.meshgrid(fx, fy)

        argument = (2 * np.pi)**2 * ((1. / λ) ** 2 - fxx ** 2 - fyy ** 2)

        #Calculate the propagating and the evanescent (complex) modes
        tmp = np.sqrt(np.abs(argument))
        kz = torch.tensor(np.where(argument >= 0, tmp, 1j*tmp))
        self.register_buffer('phase', torch.exp(1j * kz * dz))

    def forward(self, E):
        # waves (batch, 200, 200)
        fft_c = torch.fft.fft2(E)
        c = torch.fft.fftshift(fft_c)
        angular_spectrum = torch.fft.ifft2(torch.fft.ifftshift(c * self.phase))
        return angular_spectrum

class Lens(torch.nn.Module):

    def __init__(self, focus, wl = 532e-9, N_pixels = 400, pixel_size = 20e-6):
        super(Lens, self).__init__()

        coord_limit = (N_pixels//2)*pixel_size 
        mesh = np.arange(-coord_limit, coord_limit, pixel_size)
        x, y = np.meshgrid(mesh, mesh)    
        self.register_buffer('phase', torch.tensor(np.exp(-1j*np.pi/(wl*2*focus) * (x**2 + y**2))))
        self.register_buffer('amplitude', torch.zeros([N_pixels, N_pixels], dtype = torch.float32) + 1)

    def forward(self, E):
        return E * self.amplitude * self.phase

class Fraunhofer_diffraction(torch.nn.Module):

    def __init__(self, λ = 532e-9, N_pixels = 400, pixel_size = 20e-6, dz = 0.01):
        super(Fraunhofer_diffraction, self).__init__()

        x = np.arange(N_pixels) * pixel_size
        y = np.arange(N_pixels) * pixel_size

        x_grid1,y_grid1,x_grid2,y_grid2 = np.meshgrid(x,y,x,y)

        r = np.sqrt((x_grid1 - x_grid2)**2 + (y_grid1 - y_grid2)**2 + dz**2)

        self.weights = (dz/r**2) * (1/(2*np.pi*r) + 1/(1j*λ)) * np.exp(1j*2*np.pi*r/λ)

        #Calculate the propagating and the evanescent (complex) modes
        self.register_buffer('phase', self.weights)

    def forward(self, E):
        # waves (batch, 200, 200)
        return torch.matmul(E, self.weights)
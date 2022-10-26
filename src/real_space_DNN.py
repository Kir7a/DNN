import torch
import numpy as np
from torchvision import transforms

# from src.DNN import Trainer

# def prop_count(n_layer1, n_layer2, size1, size2, dist_z, wl):
#     """
#     Расчёт множителя, отвечающего за дифракционное расхождение пучка при распространении между слоями
#     :param n_layer1: размер предыдущего слоя нейросети (n*n)
#     :param n_layer2: размер нового слоя нейросети (n*n)
#     :param size1: размер пикселя на предыдущем слое (в м)
#     :param size2: размер пикселя нового слоя (в м)
#     :param dist_z: расстояние между слоями (в м)
#     :param wl: длина волны (в м)
#     :return: тензор расстояний между точками (в м)
#     """
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     coordinates1 = torch.tensor(range(-n_layer1 // 2 + 1, n_layer1 // 2 + 1), device=device).double() * size1
#     coordinates2 = torch.tensor(range(-n_layer2 // 2 + 1, n_layer2 // 2 + 1), device=device).double() * size2
#
#     x1_grid, x2_grid = torch.meshgrid(coordinates1, coordinates2, indexing='ij')
#     dx = x1_grid - x2_grid
#     dy = dx.clone()
#     dx = dx[None, :, None, :, None]
#     dy = dy[None, None, :, None, :]
#     r_tensor = torch.sqrt(dx ** 2 + dy ** 2 + dist_z ** 2)
#
#     prop_multiplier = size1 ** 2 * dist_z / (r_tensor ** 2) * (1 / (2 * np.pi * r_tensor) + 1 / (wl * 1j)) * torch.exp(
#         2.0j * np.pi * r_tensor / wl)
#
#     return prop_multiplier
#
#
# def img_propagation(input_tensor, prop_multiplier):
#     """
#     Расчёт изображения на входе следующего слоя
#     :param input_tensor изображение на выходе предыдущего слоя
#     :param prop_multiplier множитель распространения
#     """
#     output_tensor = (prop_multiplier * input_tensor[:, :, :, None, None]).sum(dim=(1, 2))
#     return output_tensor


class real_space_DNN(torch.nn.Module):
    """
    phase only modulation
    """

    def __init__(self,
                 num_layers=5,
                 wl=532e-9,
                 N_pixels=28,
                 pixel_size=100e-6,
                 N_neurons=20,
                 neuron_size=400e-6,
                 layer_distance=0.005,
                 img_distance=0.005):
        super(real_space_DNN, self).__init__()
        self.n_layers = num_layers
        self.wl = wl
        self.N_pixels = N_pixels
        self.pixel_size = pixel_size
        self.N_neurons = N_neurons
        self.neuron_size = neuron_size
        self.img_distance = img_distance
        self.layer_distance = layer_distance
        self.input_prop_multiplier = self.prop_count(0)
        self.prop_multiplier = self.prop_count(1)
        self.phase = [
            torch.nn.Parameter(torch.from_numpy(np.random.random(size=(N_neurons, N_neurons)).astype('float32') - 0.5))
            for _ in range(num_layers)]
        for i in range(num_layers):
            self.register_parameter("phase" + "_" + str(i), self.phase[i])
        # self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, unconstrained_phase=False):
        outputs = [x]
        for n in range(self.n_layers):
            x = self.img_propagation(n, x)
            x = self.exp_j_phase(n, unconstrained_phase) * x
            outputs.append(x)
        x = self.img_propagation(self.n_layers, x)
        outputs.append(x)
        x_abs = torch.abs(x) ** 2
        # output = self.softmax(detector_region(x_abs))
        # output = Trainer.detector_region(x_abs)
        return x_abs, outputs

    def exp_j_phase(self, num_layer, unconstrained_phase=False):
        if unconstrained_phase:
            constr_phase = self.phase[num_layer]
        else:
            constr_phase = 2 * np.pi * torch.sigmoid(self.phase[num_layer])
        constr_phase = transforms.functional.resize(constr_phase[None, :, :], self.N_pixels,
                                                    transforms.InterpolationMode.NEAREST).squeeze()
        a = torch.exp(1j * constr_phase)
        return a

    def prop_count(self, num_layer):
        """
        Расчёт множителя, отвечающего за дифракционное расхождение пучка при распространении между слоями
        :param num_layer номер текущего слоя, где нулевым слоем считается исходное изображение
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        n_layer1 = n_layer2 = self.N_pixels
        size1 = size2 = self.pixel_size
        if num_layer == 0:
            dist_z = self.img_distance
        else:
            dist_z = self.layer_distance

        coordinates1 = torch.tensor(range(-n_layer1 // 2 + 1, n_layer1 // 2 + 1), device=device).double() * size1
        coordinates2 = torch.tensor(range(-n_layer2 // 2 + 1, n_layer2 // 2 + 1), device=device).double() * size2

        x1_grid, x2_grid = torch.meshgrid(coordinates1, coordinates2, indexing='ij')
        dx = x1_grid - x2_grid
        dy = dx.clone()
        dx = dx[None, :, None, :, None]
        dy = dy[None, None, :, None, :]
        r_tensor = torch.sqrt(dx ** 2 + dy ** 2 + dist_z ** 2)

        prop_multiplier = size1 ** 2 * dist_z / (r_tensor ** 2) * (
                    1 / (2 * np.pi * r_tensor) + 1 / (self.wl * 1j)) * torch.exp(2.0j * np.pi * r_tensor / self.wl)

        return prop_multiplier

    def img_propagation(self, num_layer, input_tensor):
        """
        Расчёт изображения на входе следующего слоя
        :param num_layer номер текущего слоя, где нулевым слоем считается исходное изображение
        :param input_tensor изображение на выходе предыдущего слоя
        """
        if num_layer == 0:
            output_tensor = (self.input_prop_multiplier * input_tensor[:, :, :, None, None]).sum(dim=(1, 2))
        else:
            output_tensor = (self.prop_multiplier * input_tensor[:, :, :, None, None]).sum(dim=(1, 2))
        return output_tensor

    @property
    def device(self):
        return next(self.parameters()).device


class real_space_DNN_conv(torch.nn.Module):
    """
    phase only modulation
    """

    def __init__(self,
                 num_layers=5,
                 wl=532e-9,
                 N_pixels=28,
                 pixel_size=100e-6,
                 N_neurons=20,
                 neuron_size=400e-6,
                 layer_distance=0.005,
                 img_distance=0.005,
                 device='cpu'):
        super(real_space_DNN_conv, self).__init__()
        self.device = device
        self.n_layers = num_layers
        self.wl = wl
        self.N_pixels = N_pixels
        self.pixel_size = pixel_size
        self.N_neurons = N_neurons
        self.neuron_size = neuron_size
        self.img_distance = img_distance
        self.layer_distance = layer_distance
        self.input_prop_multiplier = self.prop_count(0)
        self.prop_multiplier = self.prop_count(1)
        self.phase = [
            torch.nn.Parameter(torch.from_numpy(np.random.random(size=(N_neurons, N_neurons)).astype('float32') - 0.5))
            for _ in range(num_layers)]
        for i in range(num_layers):
            self.register_parameter("phase" + "_" + str(i), self.phase[i])
        # self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, unconstrained_phase=False):
        x = x[:,None,:,:]
        outputs = [x]
        for n in range(self.n_layers):
            x = self.img_propagation(n, x)
            x = self.exp_j_phase(n, unconstrained_phase) * x
            outputs.append(x)
        x = self.img_propagation(self.n_layers, x)
        outputs.append(x)
        x_abs = torch.abs(x) ** 2
        # output = self.softmax(detector_region(x_abs))
        # output = Trainer.detector_region(x_abs)
        return x_abs, outputs

    def exp_j_phase(self, num_layer, unconstrained_phase=False):
        if unconstrained_phase:
            constr_phase = self.phase[num_layer]
        else:
            constr_phase = 2 * np.pi * torch.sigmoid(self.phase[num_layer])
        constr_phase = transforms.functional.resize(constr_phase[None, :, :], self.N_pixels,
                                                    transforms.InterpolationMode.NEAREST).squeeze()
        a = torch.exp(1j * constr_phase)
        return a

    def prop_count(self, num_layer):
        """
        Расчёт ядра свёртки, применяемой для расчета распространения пучка при распространении между слоями
        :param num_layer номер текущего слоя, где нулевым слоем считается исходное изображение
        """

        if num_layer == 0:
            dist_z = self.img_distance
        else:
            dist_z = self.layer_distance

        coordinates = torch.tensor(range(-self.N_pixels, self.N_pixels + 1),
                                   device=self.device, dtype=torch.double) * self.pixel_size
        x_grid, y_grid = torch.meshgrid(coordinates, coordinates, indexing='ij')

        r = torch.sqrt(x_grid ** 2 + y_grid ** 2 + dist_z ** 2)
        prop_multiplier = self.pixel_size ** 2 * dist_z /\
                          (r ** 2) * (1 / (2 * np.pi * r) + 1 / (self.wl * 1j)) *\
                          torch.exp(2.0j * np.pi * r / self.wl)

        return prop_multiplier[None, None, :, :]

    def img_propagation(self, num_layer, input_tensor):
        """
        Расчёт изображения на входе следующего слоя
        :param num_layer номер текущего слоя, где нулевым слоем считается исходное изображение
        :param input_tensor изображение на выходе предыдущего слоя
        """
        if num_layer == 0:
            prop_multiplier = self.input_prop_multiplier
        else:
            prop_multiplier = self.prop_multiplier

        tmp = input_tensor.type(torch.complex128).to(self.device)
        m = input_tensor.shape[-1]
        output_tensor = torch.nn.functional.conv2d(tmp, prop_multiplier, stride=1, padding=(m, m))
        return output_tensor

    # @property
    # def device(self):
    #     return next(self.parameters()).device
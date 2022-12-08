import torch
import numpy as np
import torch.nn.functional as F
from src.diffraction import DiffractiveLayer, Lens
from tqdm import tqdm
from copy import deepcopy
from torchvision import transforms

DETECTOR_POS = [
    (46, 66, 46, 66), (46, 66, 93, 113), (46, 66, 140, 160),
    (85, 105, 46, 66), (85, 105, 78, 98), (85, 105, 109, 129),
    (85, 105, 140, 160), (125, 145, 46, 66), (125, 145, 93, 113),
    (125, 145, 140, 160)
]


class Trainer():
    """
    Класс для тренировки оптической нейронной сети.
    param model: Модель нейронной сети для обучения
    param detector_pos: список позиций детекторов для классификации
    param padding: число нулевых пикселей, которое нужно добавить к изображению на вход нейронной сети.
        Если число пикселей изображения совпадает с числом пикселей в фазовой маске нейронной сети, то padding = 0
    param device: где будет проходить обучение сети 'cpu'/'cuda'
    """

    def __init__(self, model, detector_pos=DETECTOR_POS, padding=58, device='cpu'):
        self.detector_pos = detector_pos
        self.model = model
        self.padding = padding
        self.device = device

    def detector_region(self, x):
        """
        Подсчет интенсивности, которая приходится на каждый детектор в конце оптической нейронной сети.
        param x: распределение интенсивности на выходе нейронной сети
        return: тензор с суммарной интенсивностью, приходящейся на каждый детектор
        """
        detectors_list = []
        full_int = x.sum(dim=(1, 2))
        for det_x0, det_x1, det_y0, det_y1 in self.detector_pos:
            detectors_list.append(
                (x[:, det_x0: det_x1 + 1, det_y0: det_y1 + 1].sum(dim=(1, 2)) / full_int).unsqueeze(-1))
        return torch.cat(detectors_list, dim=1)

    def epoch_step(self, batch, unconstrain_phase=False):
        """
        Обработка одного батча в процессе тренировки.
        param batch: (imgs, labels) батч с изображениями и их метками
        """
        images, labels = batch
        images = images.to(self.device)
        images = F.pad(images, pad=(self.padding, self.padding, self.padding, self.padding))
        labels = labels.to(self.device)

        out_img, _ = self.model(images, unconstrain_phase)

        out_label = self.detector_region(out_img)
        _, predicted = torch.max(out_label.data, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        loss = self.loss_function(out_img, labels)
        return loss, correct, total

    def train(self,
              loss_function,
              optimizer,
              trainloader,
              testloader,
              epochs=10,
              discrete_thickness=None,
              unconstrain_phase=False):
        """
        Функция для тренировки сети.
        """
        hist = {'train loss': [],
                'test loss': [],
                'train accuracy': [],
                'test accuracy': []}
        best_acc = 0
        self.loss_function = loss_function
        for epoch in range(epochs):
            ep_loss = 0
            self.model.train()
            correct = 0
            total = 0
            for batch in tqdm(trainloader):
                # округление фазы при дискретной толщине слоёв маски
                if discrete_thickness and self.model.mask_layers[0].n:
                    self.model.round_phase(discrete_thickness)

                loss, batch_correct, batch_total = self.epoch_step(batch, unconstrain_phase)
                ep_loss += loss.item()
                correct += batch_correct
                total += batch_total

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            hist['train loss'].append(ep_loss / len(trainloader))
            hist['train accuracy'].append(correct / total)

            ep_loss = 0
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in tqdm(testloader):
                    loss, batch_correct, batch_total = self.epoch_step(batch, unconstrain_phase)
                    ep_loss += loss.item()
                    correct += batch_correct
                    total += batch_total
            hist['test loss'].append(ep_loss / len(testloader))
            hist['test accuracy'].append(correct / total)

            if hist['test accuracy'][-1] > best_acc:
                best_acc = hist['test accuracy'][-1]
                best_model = deepcopy(self.model)

            print(
                f"\nEpoch={epoch + 1} train loss={hist['train loss'][epoch]:.4}, test loss={hist['test loss'][epoch]:.4}")
            print(f"train acc={hist['train accuracy'][epoch]:.4}, test acc={hist['test accuracy'][epoch]:.4}")
            print("-----------------------")

        return hist, best_model

    def validate(self, dataloader, unconstrain_phase=False):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                images = images.to(self.device)
                images = F.pad(torch.squeeze(images), pad=(self.padding, self.padding, self.padding, self.padding))
                labels = labels.to(self.device)

                out_img, _ = self.model(images, unconstrain_phase)

                out_label = self.detector_region(out_img)
                _, predicted = torch.max(out_label.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return correct / total


class MaskLayer(torch.nn.Module):
    """
    Класс для амплитудно-фазовой маски
    """

    # сюда следует добавить возможность дискретизации толщин пикселей маски
    def __init__(self, distance_before_mask=None, wl=532e-9, N_pixels=400, pixel_size=20e-6, N_neurons=20,
                 include_amplitude=False, n=None):
        """
        :param distance_before_mask: расстояние, которое проходит излучение до маски
        :param wl: длина волны излучения, для многоканальных изображений может быть задана списком соответствующей длины
        :param N_pixels: число пикселей в изображении
        :param pixel_size: размер одного пикселя
        :param N_neurons: число нейронов (пикселей в маске)
        :param include_amplitude: применять ли амплитудную модуляцию
        :param n: комплексный показатель преломления, для многоканальных изображений - список, len(n) = len(wl)
        """
        super(MaskLayer, self).__init__()

        self.diffractive_layer = None
        if distance_before_mask is not None:
            self.diffractive_layer = DiffractiveLayer(wl, N_pixels, pixel_size, distance_before_mask)

        wl_size = 1
        try:
            wl_size = len(wl)
        except TypeError:
            wl = [wl]
        finally:
            self.register_buffer('wl', torch.tensor(wl, dtype=torch.float32))

        n_size = 1
        try:
            n_size = len(n)
        except TypeError:
            n = [n]
        finally:
            if n_size == wl_size or n_size == 1:
                self.register_buffer('n', torch.tensor(n, dtype=torch.complex64))
            else:
                raise Exception("Numbers of wl does not match number of n")

        self.phase = torch.nn.Parameter(torch.zeros([wl_size, N_neurons, N_neurons], dtype=torch.float32))
        if include_amplitude and (n is None):
            self.amplitude = torch.nn.Parameter(torch.zeros([wl_size, N_neurons, N_neurons], dtype=torch.float32) + 1)
        self.phase_amp_mod = include_amplitude
        self.N_pixels = N_pixels
        self.pixel_size = pixel_size
        self.N_neurons = N_neurons
        # self.n = n

    def forward(self, E, unconstrain_phase=False):
        out = E
        if self.diffractive_layer is not None:
            out = self.diffractive_layer(out)
        if unconstrain_phase:
            constr_phase = self.phase
        else:
            constr_phase = 2 * np.pi * torch.sigmoid(self.phase)
        modulation = torch.cos(constr_phase) + 1j * torch.sin(constr_phase)
        if self.phase_amp_mod:
            if self.n is None:
                constr_amp = F.relu(self.amplitude) / F.relu(self.amplitude).max()
            else:
                constr_amp = torch.exp(
                    - np.imag(self.n[:, None, None]) * 2 * np.pi / self.wl[:, None, None] * self.calc_thickness(
                        constr_phase))
            modulation = constr_amp * modulation
        modulation = transforms.functional.resize(modulation.real, self.N_pixels, transforms.InterpolationMode.NEAREST)\
            + 1j * transforms.functional.resize(modulation.imag, self.N_pixels, transforms.InterpolationMode.NEAREST)
        out = modulation * out
        return out

    def calc_thickness(self, phase=None):
        """
        Вычисление толщины, соответствующей набегу фаз в маске
        """
        if phase is None:
            phase = 2 * np.pi * torch.sigmoid(self.phase)
        thickness = self.wl[:, None, None] * phase / (2 * np.pi * np.real(self.n[:, None, None]))
        return thickness


class DNN(torch.nn.Module):
    """
    phase only modulation
    """

    def __init__(self, num_layers=5, wl=532e-9, N_pixels=400, pixel_size=20e-6, distance=0.01):

        super(DNN, self).__init__()

        self.phase = [
            torch.nn.Parameter(torch.from_numpy(np.random.random(size=(N_pixels, N_pixels)).astype('float32') - 0.5))
            for _ in range(num_layers)]
        for i in range(num_layers):
            self.register_parameter("phase" + "_" + str(i), self.phase[i])
        self.diffractive_layers = torch.nn.ModuleList(
            [DiffractiveLayer(wl, N_pixels, pixel_size, distance) for _ in range(num_layers)])
        self.last_diffractive_layer = DiffractiveLayer(wl, N_pixels, pixel_size, distance)

    def forward(self, x):
        # x (batch, N_pixels, N_pixels)
        for index, layer in enumerate(self.diffractive_layers):
            temp = layer(x)
            # constr_phase = self.phase[index]#
            constr_phase = 2 * np.pi * torch.sigmoid(self.phase[index])
            exp_j_phase = torch.exp(1j * constr_phase)  # torch.cos(constr_phase)+1j*torch.sin(constr_phase)
            x = temp * exp_j_phase
        x = self.last_diffractive_layer(x)
        x_abs = torch.abs(x) ** 2
        output = self.detector_region(x_abs)
        return output, x_abs


# Архитектура Фурье дифракционной сети
class Fourier_DNN(torch.nn.Module):
    """
    phase only modulation
    """

    def __init__(self, num_layers=5, wl=532e-9, N_pixels=400, pixel_size=20e-6, distance=0.01, lens_focus=10e-2):

        super(Fourier_DNN, self).__init__()

        # self.phase = [torch.nn.Parameter(torch.from_numpy(np.random.random(size=(N_pixels, N_pixels)).astype('float32')-0.5)) for _ in range(num_layers)]
        self.phase = [torch.nn.Parameter(torch.from_numpy(np.zeros((N_pixels, N_pixels)).astype('float32'))) for _ in
                      range(num_layers)]
        for i in range(num_layers):
            self.register_parameter("phase" + "_" + str(i), self.phase[i])

        coord_limit = (N_pixels // 2) * pixel_size
        mesh = np.arange(-coord_limit, coord_limit, pixel_size)
        x, y = np.meshgrid(mesh, mesh)
        self.lens_phase = torch.tensor(np.exp(-1j * np.pi / (wl * 2 * lens_focus) * (x ** 2 + y ** 2)))
        self.first_diffractive_layer = DiffractiveLayer(wl, N_pixels, pixel_size, lens_focus - distance)
        self.diffractive_layers = torch.nn.ModuleList(
            [DiffractiveLayer(wl, N_pixels, pixel_size, distance) for _ in range(0, num_layers)])
        self.double_f_layer = DiffractiveLayer(wl, N_pixels, pixel_size, 2 * lens_focus)
        self.single_f_layer = DiffractiveLayer(wl, N_pixels, pixel_size, lens_focus)

    def forward(self, x):
        # x (batch, 200, 200)
        outputs = [x]
        # x = self.double_f_layer(x)
        x = self.single_f_layer(x)
        outputs.append(x)
        x = x * self.lens_phase
        x = self.first_diffractive_layer(x)
        for index, layer in enumerate(self.diffractive_layers):
            temp = layer(x)
            outputs.append(x)
            constr_phase = np.pi * torch.sigmoid(self.phase[index])
            exp_j_phase = torch.exp(1j * constr_phase)
            x = temp * exp_j_phase
        x = self.single_f_layer(x)
        outputs.append(x)
        x = x * self.lens_phase
        # x = self.double_f_layer(x)
        x = self.single_f_layer(x)
        outputs.append(x)
        # x_abs (batch, 200, 200)
        x_abs = torch.abs(x) ** 2
        # output = self.detector_region(x_abs)
        # return output, x_abs, outputs
        return x_abs, outputs


class new_Fourier_DNN(torch.nn.Module):
    """
    Fourier Diffractive Neural Network
    """

    def __init__(self,
                 num_layers=5,
                 wl=532e-9,
                 N_pixels=200,
                 pixel_size=10e-6,
                 N_neurons=40,
                 distance=5e-3,
                 lens_focus=100e-3,
                 include_amplitude_modulation=True,
                 dn=None):
        super(new_Fourier_DNN, self).__init__()
        self.lens_diffractive_layer = DiffractiveLayer(wl, N_pixels, pixel_size, lens_focus)
        self.lens = Lens(lens_focus, wl, N_pixels, pixel_size)
        self.first_diffractive_layer = DiffractiveLayer(wl, N_pixels, pixel_size, lens_focus - distance)

        self.mask_layers = torch.nn.ModuleList([MaskLayer(distance_before_mask=distance,
                                                          wl=wl,
                                                          N_pixels=N_pixels,
                                                          pixel_size=pixel_size,
                                                          N_neurons=N_neurons,
                                                          include_amplitude=include_amplitude_modulation,
                                                          n=dn) for _ in range(0, num_layers)])

    def forward(self, E, unconsrtain_phase=False):
        outputs = [E]
        E = self.lens_diffractive_layer(E)
        E = self.lens(E)
        E = self.first_diffractive_layer(E)
        outputs.append(E)
        for layer in self.mask_layers:
            E = layer(E, unconsrtain_phase)
            outputs.append(E)
        E = self.lens_diffractive_layer(E)
        E = self.lens(E)
        outputs.append(E)
        E = self.lens_diffractive_layer(E)
        E_abs = torch.abs(E) ** 2
        return E_abs.sum(dim=1), outputs

    def round_phase(self, thick_discr):
        phase_discr = (self.mask_layers[0].n.real) * thick_discr * 2 * np.pi / self.mask_layers[0].wl
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.copy_(
                    torch.round(param.clone().detach() / phase_discr[:, None, None]) * phase_discr[:, None, None])

    @property
    def device(self):
        return next(self.parameters()).device

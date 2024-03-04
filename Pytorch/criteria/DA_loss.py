import torch
from torch import nn
import torch.nn.functional as F

from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
)
class Net(nn.Module):
    def __init__(self, encoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:])  # relu3_1 -> relu4_1

        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(5):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, input, alpha=1.0):
        assert 0 <= alpha <= 1
        feat = self.encode(input)

        # print(feat.size())
        # raise RuntimeError
        return feat
class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, dropout=0.5, use_shot=True):
        super(feat_bootleneck, self).__init__()
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=dropout)
        self.bottleneck.apply(init_weights)
        self.use_shot = use_shot

    def forward(self, x):
        x = self.bottleneck(x)
        if self.use_shot:
            x = self.bn(x)
            x = self.dropout(x)
        return x
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
class Classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, use_shot=True):
        super(Classifier, self).__init__()
        self.bottleneck = feat_bootleneck(7*7*512)
        self.fc = nn.Linear(bottleneck_dim, class_num, bias=False)
        self.fc.apply(init_weights)
    
    def forward(self, x):
        
        feat = self.bottleneck(x.view(x.size()[0],-1))
        x = self.fc(feat)
        return x
class KDLoss(nn.Module):
    def __init__(self, temperature=1):
        super(KDLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_output, teacher_output):
        """
        NOTE: the KL Divergence for PyTorch comparing the prob of teacher and log prob of student,
        mimicking the prob of ground truth (one-hot) and log prob of network in CE loss
        """
        # x -> input -> log(q)
        log_q = F.log_softmax(student_output / self.temperature, dim=1)
        # y -> target -> p
        p = F.softmax(teacher_output / self.temperature, dim=1)
        # F.kl_div(x, y) -> F.kl_div(log_q, p)
        # l_n = y_n \cdot \left( \log y_n - x_n \right) = p * log(p/q)
        l_kl = F.kl_div(log_q, p)  # forward KL
        return l_kl
class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, featmap_src_T, featmap_tgt_S):
        B, C, H, W = featmap_src_T.shape
        f_src, f_tgt = featmap_src_T.view([B, C, H * W]), featmap_tgt_S.view([B, C, H * W])
        # calculate Gram matrices
        A_src, A_tgt = torch.bmm(f_src, f_src.transpose(1, 2)), torch.bmm(f_tgt, f_tgt.transpose(1, 2))
        A_src, A_tgt = A_src / (H * W), A_tgt / (H * W)
        loss = F.mse_loss(A_src, A_tgt)
        return loss
class BatchSimLoss(nn.Module):
    def __init__(self):
        super(BatchSimLoss, self).__init__()

    def forward(self, featmap_src_T, featmap_tgt_S):
        B, C, H, W = featmap_src_T.shape
        f_src, f_tgt = featmap_src_T.view([B, C * H * W]), featmap_tgt_S.view([B, C * H * W])
        A_src, A_tgt = f_src @ f_src.T, f_tgt @ f_tgt.T
        A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
        loss_batch = torch.norm(A_src - A_tgt) ** 2 / B
        return loss_batch
class PixelSimLoss(nn.Module):
    def __init__(self):
        super(PixelSimLoss, self).__init__()

    def forward(self, featmap_src_T, featmap_tgt_S):
        B, C, H, W = featmap_src_T.shape
        loss_pixel = 0
        for b in range(B):
            f_src, f_tgt = featmap_src_T[b].view([C, H * W]), featmap_tgt_S[b].view([C, H * W])
            A_src, A_tgt = f_src.T @ f_src, f_tgt.T @ f_tgt
            A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
            loss_pixel += torch.norm(A_src - A_tgt) ** 2 / (H * W)
        loss_pixel /= B
        return loss_pixel
class ChannelSimLoss(nn.Module):
    def __init__(self):
        super(ChannelSimLoss, self).__init__()

    def forward(self, featmap_src_T, featmap_tgt_S):
        # B, C, H, W = featmap_src_T.shape
        # print(featmap_src_T.size())
        B, C, H, W = featmap_src_T.size()
        loss = 0
        for b in range(B):
            f_src, f_tgt = featmap_src_T[b].view([C, H * W]), featmap_tgt_S[b].view([C, H * W])
            A_src, A_tgt = f_src @ f_src.T, f_tgt @ f_tgt.T
            A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
            loss += torch.norm(A_src - A_tgt) ** 2 / C
            # loss += torch.norm(A_src - A_tgt, p=1)
        loss /= B
        return loss
class ChannelSimLoss1D(nn.Module):
    def __init__(self):
        super(ChannelSimLoss1D, self).__init__()

    def forward(self, feat_src_T, feat_tgt_S):
        B, C = feat_src_T.shape
        loss = torch.zeros([]).cuda()
        for b in range(B):
            f_src, f_tgt = feat_src_T[b].view([C, 1]), feat_tgt_S[b].view([C, 1])
            A_src, A_tgt = f_src @ f_src.T, f_tgt @ f_tgt.T
            A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
            loss += torch.norm(A_src - A_tgt) ** 2 / C
            # loss += torch.norm(A_src - A_tgt, p=1)
        loss /= B
        return loss

class DAloss(nn.Module):
    def __init__(self):
        super(DAloss, self).__init__()
        print('Loading DA model')
        self.modelS = Net(vgg)
        self.modelT = Net(vgg)
        self.modelC = Classifier(12)

        Spath ="./pretrained_models/source_model.pth"
        Tpath ="./pretrained_models/target_model.pth"
        Cpath ="./pretrained_models/Classifier_model.pth"

        self.modelS.load_state_dict(torch.load(Spath))
        self.modelT.load_state_dict(torch.load(Tpath))
        self.modelC.load_state_dict(torch.load(Cpath))
        self.modelS.eval()
        self.modelT.eval()
        self.modelC.eval()

        self.batch_loss = BatchSimLoss()
        self.pixel_loss = PixelSimLoss()
        self.style_loss = StyleLoss()
        self.channel_loss = ChannelSimLoss() #if opts.use_channel else StyleLoss()
        self.channel_loss_1d = ChannelSimLoss1D()
        self.KDLoss = KDLoss()
        
    def forward(self, y_hat, x):
        y_hat_hat = F.interpolate(y_hat, size=(224,224))
        x_hat = F.interpolate(x, size=(224,224))
        featS = self.modelS(y_hat_hat)
        featT = self.modelT(x_hat)

        loss_channel = self.channel_loss(featS, featT)
        loss_batch = self.batch_loss(featS, featT)
        loss_pixel = self.pixel_loss(featS, featT)

        predS = self.modelC(featS)
        predT = self.modelC(featT)
        loss_KD = self.KDLoss(predS,predT)

        return loss_channel,loss_batch,loss_pixel,loss_KD

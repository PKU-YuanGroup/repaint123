import torch

from contextual_loss import functional as F
from contextual_loss import ContextualBilateralLoss

test_shape = [1, 256, 48, 48]


def test_module():
    prediction = torch.rand(*test_shape)
    f = ContextualBilateralLoss()
    loss = f(prediction, prediction)
    assert loss.shape == torch.Size([])


def test_module_gpu():
    prediction = torch.rand(*test_shape).to('cuda:0')
    f = ContextualBilateralLoss().to('cuda:0')
    loss = f(prediction, prediction)
    assert loss.shape == torch.Size([])


def test_vgg():
    prediction = torch.rand(test_shape[0], 3, *test_shape[2:])
    f = ContextualBilateralLoss(use_vgg=True)
    loss = f(prediction, prediction)
    assert loss.shape == torch.Size([])


def test_vgg_gpu():
    prediction = torch.rand(test_shape[0], 3, *test_shape[2:]).to('cuda:0')
    f = ContextualBilateralLoss(use_vgg=True).to('cuda:0')
    loss = f(prediction, prediction)
    assert loss.shape == torch.Size([])


def test_cosine():
    prediction = torch.rand(*test_shape)
    loss = F.contextual_bilateral_loss(
        prediction, prediction, loss_type='cosine')
    assert loss.shape == torch.Size([])


def test_cosine_gpu():
    prediction = torch.rand(*test_shape).to('cuda:0')
    loss = F.contextual_bilateral_loss(
        prediction, prediction, loss_type='cosine')
    assert loss.shape == torch.Size([])


def test_l1():
    prediction = torch.rand(*test_shape)
    loss = F.contextual_bilateral_loss(
        prediction, prediction, loss_type='l1')
    assert loss.shape == torch.Size([])


def test_l1_gpu():
    prediction = torch.rand(*test_shape).to('cuda:0')
    loss = F.contextual_bilateral_loss(
        prediction, prediction, loss_type='l1')
    assert loss.shape == torch.Size([])


def test_l2():
    prediction = torch.rand(*test_shape)
    loss = F.contextual_bilateral_loss(
        prediction, prediction, loss_type='l2')
    assert loss.shape == torch.Size([])


def test_l2_gpu():
    prediction = torch.rand(*test_shape).to('cuda:0')
    loss = F.contextual_bilateral_loss(
        prediction, prediction, loss_type='l2')
    assert loss.shape == torch.Size([])

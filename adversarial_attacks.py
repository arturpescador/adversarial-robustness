import torch
import torch.nn.functional as F

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def pgd_attack(model, images, labels, epsilon, alpha, num_iter):
    perturbed_images = images.clone().detach().to(images.device)
    perturbed_images.requires_grad = True

    for i in range(num_iter):
        outputs = model(perturbed_images)
        loss = F.nll_loss(outputs, labels)
        model.zero_grad()
        loss.backward()

        perturbed_images = perturbed_images + alpha * perturbed_images.grad.sign()
        perturbed_images = torch.max(torch.min(perturbed_images, images + epsilon), images - epsilon)
        perturbed_images = torch.clamp(perturbed_images, 0, 1)

        # for the next iteration
        perturbed_images = perturbed_images.detach().clone()
        perturbed_images.requires_grad = True

    return perturbed_images


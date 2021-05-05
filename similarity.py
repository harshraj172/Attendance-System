# -*- coding: utf-8 -*-
from scipy.spatial import distance

def similarity_score(ref_image, selfie_image, device):
    """
    Given a selfie image and the reference image to compare.
    Returns the similarity score.
    ref_image: Image stored with company for reference.
    selfie_image: Image taken by employee to mark his presence.
    device : "cuda" or "cpu" device.
    """
    # for ref_image
    ref_image_tensor = T.ToTensor()(ref_image)
    ref_image_tensor = ref_image_tensor.unsqueeze(0)
    
    # for selfie_image
    selfie_image_tensor = T.ToTensor()(selfie_image)
    selfie_image_tensor = selfie_image_tensor.unsqueeze(0)

    with torch.no_grad():
        ref_image_embedding = encoder(ref_image_tensor).cpu().detach().numpy()
        selfie_image_embedding = encoder(selfie_image_tensor).cpu().detach().numpy()
        
    ref_flattened_embedding = image_embedding.reshape((ref_image_embedding.shape[0], -1))
    selfie_flattened_embedding = image_embedding.reshape((selfie_image_embedding.shape[0], -1))

    cosine_score = distance.cosine(ref_flattened_embedding, selfie_flattened_embedding)
    return cosine_score

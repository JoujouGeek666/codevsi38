import numpy as np
from PIL import Image
from skimage.feature import graycomatrix
from skimage.metrics import structural_similarity 
import cv2
import pypiqe
import matplotlib.pyplot as plt

def autocorrelation_x_normalized(image_path, shift):
    # Charger l'image
    image = Image.open(image_path)
    
    # Conversion de l'image en niveaux de gris
    gray_image = image.convert('L')
    # Conversion de l'image en tableau numpy
    gray_array = np.array(gray_image, dtype=np.float32)

    # Décalage de l'image horizontalement
    shifted_array = np.roll(gray_array, shift, axis=1)
        
    # Calcul de la corrélation entre l'image d'origine et l'image décalée
    correlation = np.mean(gray_array * shifted_array)

    # Normalisation en divisant par la variance de l'image
    correlation_normalized = correlation / np.var(gray_array)

    return correlation_normalized

def image_correlation(image1, image2):
    # Redimensionner les images si nécessaire
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    # Calculer le coefficient de corrélation
    correlation_coefficient = np.corrcoef(image1.flatten(), image2.flatten())[0, 1]
    return correlation_coefficient

def image_mean_squared_error(image1, image2):
    # Redimensionner les images si nécessaire
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    # Calculer l'erreur quadratique moyenne
    mse = np.mean((image1 - image2) ** 2)
    return mse

def image_energy_contour(image):
    # Calculer les gradients horizontaux et verticaux
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculer l'énergie des gradients en utilisant la somme des carrés des gradients
    energy_matrix = np.sqrt(gradient_x**2 + gradient_y**2)
    energy_global = np.sum(energy_matrix)
    return energy_matrix, energy_global

def image_contrast(image):
    # Initialiser le contraste à 0
    contrast = 0
    # Calculer la matrice de cooccurrence des niveaux de gris
    glcm = image_glcm(image)
    # Calculer le contraste
    for i in range(len(glcm)):
        for j in range(len(glcm)):
            contrast += glcm[i][j]*((i-j)**2)
    return contrast

def image_glcm(image):
    # Calculer la matrice de cooccurrence des niveaux de gris
    glcm = graycomatrix(image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    return glcm

def image_homogeneity(image):
    # Initialiser l'homogénéité à 0
    homogeneity = 0
    # Calculer la matrice de cooccurrence des niveaux de gris
    glcm = image_glcm(image)
    # Calculer l'homogénéité
    for i in range(len(glcm)):
        for j in range(len(glcm)):
            homogeneity += glcm[i][j] / (1 + (i-j)**2)
    return homogeneity

def image_ssim(image1, image2):
    # Redimensionner les images si nécessaire
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    # Convertir les images en tableaux numpy
    image1_np = np.array(image1)
    image2_np = np.array(image2)
    # Calculer le SSIM
    ssim_value, _ = structural_similarity(image1_np, image2_np, full=True)
    return ssim_value

def image_piqe(image):
    # Calculer le score PIQE
    score, activity_mask, noticeable_artifact_mask, noise_mask  = pypiqe.piqe(image)   
    return score

def image_offset_dependent_correlation(image1, image2, max_offset=30):
    # Initialiser le meilleur décalage et la meilleure corrélation
    best_offset = (0, 0)
    best_correlation = -1
    # Parcourir tous les décalages possibles
    for dy in range(-max_offset, max_offset + 1):
        for dx in range(-max_offset, max_offset + 1):
            offset_image2 = np.roll(image2, shift=(dy, dx), axis=(0, 1))
            correlation = image_correlation(image1, offset_image2)
            if correlation > best_correlation:
                best_correlation = correlation
                best_offset = (dy, dx)
    return best_offset, best_correlation

def image_cumulative_histogram(image):
    # Convertir l'image en tableau numpy
    image_np = np.array(image)
    # Calculer l'histogramme avec un nombre élevé de bins pour une meilleure résolution
    values, base = np.histogram(image_np, bins=256, range=(0, 256))
    # Calculer le cumulatif
    cumulative = np.cumsum(values)
    # Afficher la fonction cumulative
    plt.plot(base[:-1], cumulative, c='blue')
    plt.xlim(0, 255)  # Définir la limite de l'axe x pour couvrir toute la plage des valeurs de pixels
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Cumulative Frequency')
    plt.title('Cumulative Histogram')
    plt.show()

def image_cumulative_histogram_zoom(image):
    # Convertir l'image en tableau numpy
    image_np = np.array(image)
    # Calculer l'histogramme
    values, base = np.histogram(image_np, bins=40)
    # Calculer le cumulatif
    cumulative = np.cumsum(values)
    # Afficher la fonction cumulative
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Cumulative Frequency')
    plt.title('Cumulative Histogram')
    plt.plot(base[:-1], cumulative, c='blue')
    plt.show()

def correlations(images):
    # Initialiser les corrélations
    print("Correlations entre les 15 images : \n")
    correlations = np.zeros((5, 3, 3))
    for f in range(len(images)):
        for i in range(len(images[f])):
            for j in range(i+1, len(images[f])):
                image1 = cv2.imread(images[f][i], cv2.IMREAD_GRAYSCALE)
                image2 = cv2.imread(images[f][j], cv2.IMREAD_GRAYSCALE)
                correlations[f, i, j] = image_correlation(image1, image2)
                correlations[f, j, i] = correlations[f, i, j]
    return correlations

def mean_squared_errors(images):
    # Initialiser les erreurs quadratiques moyennes
    mse = np.zeros((5, 3, 3))
    for f in range(len(images)):
        for i in range(len(images[f])):
            for j in range(i+1, len(images[f])):
                image1 = cv2.imread(images[f][i], cv2.IMREAD_GRAYSCALE)
                image2 = cv2.imread(images[f][j], cv2.IMREAD_GRAYSCALE)
                mse[f, i, j] = image_mean_squared_error(image1, image2)
                mse[f, j, i] = mse[f, i, j]
    return mse

def energies(images):
    # Initialiser les énergies
    energies = np.zeros((5, 3))
    for f in range(len(images)):
        for i in range(len(images[f])):
            image = cv2.imread(images[f][i], cv2.IMREAD_GRAYSCALE)
            energie = image_energy_contour(image)[0]
            energies[f, i] = image_energy_contour(image)[1]
    return energies

def autocorrelations(images):
    # Initialiser les autocorrélations
    print("Les autocorrélations des 15 images :")
    autocorrelations = np.zeros((5, 3))
    shift = 10
    for f in range(len(images)):
        for i in range(len(images[f])):
            autocorrelations[f, i] = autocorrelation_x_normalized(images[f][i], shift)
    return autocorrelations

def contrasts(images):
    # Initialiser les contrastes
    contrasts = np.zeros((5, 3))
    for f in range(len(images)):
        for i in range(len(images[f])):
            image = cv2.imread(images[f][i], cv2.IMREAD_GRAYSCALE)
            contrasts[f, i] = image_contrast(image)[0, 0]
    return contrasts

def homogeneities(images):
    # Initialiser les homogénéités
    homogeneities = np.zeros((5, 3))
    for f in range(len(images)):
        for i in range(len(images[f])):
            image = cv2.imread(images[f][i], cv2.IMREAD_GRAYSCALE)
            homogeneities[f, i] = image_homogeneity(image)[0, 0]
    return homogeneities

def ssim(images):
    # Initialiser les SSIM
    ssim = np.zeros((5, 3, 3))
    for f in range(len(images)):
        for i in range(len(images[f])):
            for j in range(i+1, len(images[f])):
                image1 = cv2.imread(images[f][i], cv2.IMREAD_GRAYSCALE)
                image2 = cv2.imread(images[f][j], cv2.IMREAD_GRAYSCALE)
                ssim[f, i, j] = image_ssim(image1, image2)
                ssim[f, j, i] = ssim[f, i, j]
    return ssim

def piqe(images):
    # Initialiser les scores PIQE
    piqe = np.zeros((5, 3))
    for f in range(len(images)):
        for i in range(len(images[f])):
            image = cv2.imread(images[f][i])
            piqe[f, i] = image_piqe(image)
    return piqe

def odc(images):
    # Initialiser les corrélations dépendantes des décalages
    correlations = np.zeros((5, 3, 3))
    for f in range(len(images)):
        for i in range(len(images[f])):
            for j in range(i+1, len(images[f])):
                image1 = cv2.imread(images[f][i])
                image2 = cv2.imread(images[f][j])
                correlations[f, i, j] = image_offset_dependent_correlation(image1, image2)[1]
                correlations[f, j, i] = correlations[f, i, j]
    return correlations

def histograms(images):
    for f in range(len(images)):
        for i in range(len(images[f])):
            image = cv2.imread(images[f][i], cv2.IMREAD_GRAYSCALE)
            image_cumulative_histogram(image, f"Plots/filtre_{f+1}_type_{i+1}.png")

def histos(images):
    # Créer une figure avec des sous-graphes
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    axs = axs.flatten()
    # Définir les couleurs et les styles de lignes
    colors = ['blue', 'green', 'red']
    linestyles = ['solid', 'dotted', 'dashed']
    
    for i in range(5):
        for j in range(3):
            image = cv2.imread(images[i][j], cv2.IMREAD_GRAYSCALE)
            image_np = np.array(image)
            values, base = np.histogram(image_np, bins=40)
            cumulative = np.cumsum(values)
            axs[i].plot(base[:-1], cumulative, color=colors[j], linestyle=linestyles[j], label=f'Image {j+1}')

        axs[i].set_title(f'Filtre : {i+1}')
        axs[i].set_xlabel('Pixels intensity')
        axs[i].set_ylabel('Cumulative frequency')
        axs[i].legend()

    fig.delaxes(axs[-1])  # Supprimer le dernier subplot inutilisé

    plt.tight_layout()  # Ajuster les espacements
    plt.show()

def histos_zoom(images, natures):
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))  # Ajustement de la taille de la figure
    axs = axs.flatten()  # Pour itérer facilement sur les subplots

    colors = ['blue', 'green', 'red']  
    linestyles = ['solid', 'dotted', 'dashed']
    

    for i in range(5): 
        for j in range(3):
            image = cv2.imread(images[i][j], cv2.IMREAD_GRAYSCALE)
            image_np = np.array(image)
            values, base = np.histogram(image_np, bins=40)
            cumulative = np.cumsum(values)
            cumulative_norm = np.array(cumulative)/cumulative[-1]
            axs[i].plot(base[:-1], cumulative_norm, color=colors[j], linestyle=linestyles[j], label=f'Image muscle {natures[j]}')

        axs[i].set_title(f'Filtre {i+1}')
        axs[i].set_xlabel('Pixels intensity')
        axs[i].set_ylabel('Cumulative frequency')
        axs[i].legend()

    fig.delaxes(axs[-1])  # Suppression du dernier subplot inutilisé

    plt.tight_layout()  # Ajustement des espacements
    plt.show()

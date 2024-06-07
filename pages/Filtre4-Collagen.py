import streamlit as st
import statData as data

# Configuration de la page
st.set_page_config(
    page_title="Caractérisation d'images de muscles sains, malades et traités",
    layout="centered",
    initial_sidebar_state="auto"
)


# Choix du filtre dans la barre latérale
sidebar_reinitialize_button = st.sidebar.button("Rénitialiser le formulaire")
if 'reset' not in st.session_state or sidebar_reinitialize_button or st.session_state.pageNumber != 4:
    # Vider les parametres de la session si reinitialisation
    for key in st.session_state:
        if key != 'selected_mode' and key != 'filter' and key != 'step' and key != 'reset':
            del st.session_state[key]
    st.session_state.reset = True
    st.session_state.filter = 'Filtre 4 : Collagen'
    st.session_state.pageNumber = 4

# Initialisation ou réinitialisation des étapes
if 'reset' in st.session_state and st.session_state.reset:
    st.session_state.step = 1
    st.session_state.reset = False

# Affichage conditionnel basé sur l'étape
if st.session_state.step == 1:
    st.title("Voudriez-vous :")
    selected_mode = st.radio(
        label='',
        options=["Consulter les données propres à une image en particulier",
                "Consulter les données croisées entre deux images"]
    )

    if st.button("Suivant"):
        st.session_state.selected_mode = selected_mode
        st.session_state.step = 2
        st.experimental_rerun()

# Affichage des choix d'images
if st.session_state.step == 2 and 'selected_mode' in st.session_state:
    
    st.title("Choisissez les images :")

    # Liste des images disponibles avec description
    images = {
        "Image Muscle Sain":"Ressources/GR_420-480_fluo_000.tif",
        "Image Muscle Malade":"Ressources/GRMD_420-480_fluo_000.tif",
        "Image Muscle Traité":"Ressources/GRMDT_420-480_fluo_000.tif"
    }

    if st.session_state.selected_mode == "Consulter les données propres à une image en particulier":
        # Interface pour la sélection d'une seule image
        selected_image = st.radio("Sélectionnez une image pour l'analyse :", list(images.keys()))

        # Mémorisation de la sélection
        st.session_state.selected_image = selected_image
        st.session_state.selected_image_path = dict((key, images[key]) for key, value in images.items() if key == selected_image)
        if st.button("Analyser"):
            if selected_image:
                st.write(f"Vous avez sélectionné : {selected_image}.")
                # On reexecute le code du debut
                st.session_state.step = 3
                st.experimental_rerun()
            else:
                st.error("Veuillez sélectionner une image pour continuer.")

    elif st.session_state.selected_mode == "Consulter les données croisées entre deux images":
        # Interface pour la sélection de deux images
        selected_images = st.session_state.get('selected_images', {})
        for key, value in images.items():
            selected_images[key] = st.checkbox(key, value=selected_images.get(key, False))

        st.session_state.selected_images = selected_images
        st.session_state.selected_images_paths = dict((key, images[key]) for key, value in st.session_state.selected_images.items() if value)
        if st.button("Analyser les données croisées"):
            # Compter combien d'images ont été sélectionnées
            count_selected = sum(selected_images.values())
            if count_selected == 2:
                st.write(f"Vous avez sélectionné {count_selected} images.")
                # On reexecute le code du debut
                st.session_state.step = 4
                st.experimental_rerun()
            else:
                st.error("Veuillez sélectionner exactement deux images pour continuer.")

if st.session_state.step == 3 and 'selected_image_path' in st.session_state:
    st.title("Choisissez les statistiques à afficher:")
    stats = {"Autocorrélation" : 0, "Energie" : 1, "Contraste" : 2, "Homogénéité" : 3, "PIQE" : 4}
    # Interface pour la selection des statistiques
    selected_stats = st.session_state.get('selected_stats', {})
    for key, value in stats.items():
        selected_stats[key] = st.checkbox(key, value=selected_stats.get(key, False))
    st.session_state.selected_stats = selected_stats

    # Indexes in the data matrix
    st.session_state.selected_stats_indexes = dict((key, stats[key]) for key, value in st.session_state.selected_stats.items() if value)
    if st.button("Evaluer les statistiques"):
            # Compter combien d'images ont été sélectionnées
            count_selected = sum(selected_stats.values())
            if count_selected != 0:
                selected_stats = st.session_state.selected_stats_indexes
                selected_path = st.session_state.selected_image_path
                stats = data.getDataForImage(selected_path, selected_stats)
                for stat, value in stats.items():
                    for image_type in selected_path.keys():
                        st.write(f"**{stat}** de l'_{image_type}_ : **{value}**")
            else:
                st.error("Veuillez sélectionner au moins une statistique.")

if st.session_state.step == 4 and 'selected_images_paths' in st.session_state:
    st.title("Choisissez les statistiques à afficher:")
    stats = {"Corrélation" : 5, "Erreur quadratique moyenne" : 6, "SSIM" : 7, "ODC" : 8}
    # Interface pour la selection des statistiques
    selected_stats = st.session_state.get('selected_stats', {})
    for key, value in stats.items():
        selected_stats[key] = st.checkbox(key, value=selected_stats.get(key, False))
    st.session_state.selected_stats = selected_stats

    # Indexes in the data matrix
    st.session_state.selected_stats_indexes = dict((key, stats[key]) for key, value in st.session_state.selected_stats.items() if value)
    if st.button("Evaluer les statistiques"):
            # Compter combien d'images ont été sélectionnées
            count_selected = sum(selected_stats.values())
            if count_selected != 0:
                selected_stats = st.session_state.selected_stats_indexes
                selected_paths = st.session_state.selected_images_paths
                stats = data.getDataForImages(selected_paths, selected_stats)
                for stat, value in stats.items():
                    images_types = list(selected_paths.keys())
                    st.write(f"**{stat}** entre l'_{images_types[0]}_ et l'_{images_types[1]}_: **{value}**")
            else:
                st.error("Veuillez sélectionner au moins une statistique.")


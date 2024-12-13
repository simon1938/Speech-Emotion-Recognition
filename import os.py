import os

# Répertoire contenant les images
spectrograms_dir = "audio_representations/audio_representations_Tess/MFCCs/"

# Parcourir tous les fichiers dans le répertoire
for file_name in os.listdir(spectrograms_dir):
    if file_name.endswith(".png"):
        # Extraire la troisième donnée du nom de fichier
        try:
            third_feature = file_name.split("-")[2]
            # Vérifier si la troisième donnée est '00'
            if third_feature == "00":
                # Construire le chemin complet du fichier
                file_path = os.path.join(spectrograms_dir, file_name)
                # Supprimer le fichier
                os.remove(file_path)
                # Imprimer le nom du fichier supprimé
                print(f"Supprimé : {file_name}")
        except IndexError:
            print(f"Nom de fichier invalide : {file_name}")
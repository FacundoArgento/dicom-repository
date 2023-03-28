import dicom_library as dl


path="/home/facundo/Documents/Unnoba/Investigaciónes Patológicas/Datasets - Dicoms/CasoA/SE0004 CINE_TF2D13_RETRO_EJE_LARGO"

root_directory="/home/facundo/Documents/Unnoba/Investigaciónes Patológicas/Mascaras/PRUEBAS/Prueba sin excel/Casos Libreria"
save_directory="/home/facundo/Documents/Unnoba/Investigaciónes Patológicas/Mascaras/PRUEBAS/Prueba sin excel/Mascaras Prueba Libreria"
#dl.anonymize_files(path=path)

#dl.generate_mask_from_mat_file(root_directory=root_directory, save_directory=save_directory)

dl.build_full_mask(save_directory)
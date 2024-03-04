Zawartość pliku source_pack

    Coco_base_dataset/
        images/
        result.json

    F-RCNN/
        create_sample_images_with_trained_model.py - Wczytanie wybranego modelu i zastosowanie go do wygenerowania i zapisania masek dla zdjęć testowych
        F-RCNN base training loop.py - program zawierający logikę treningową dla modelu, wraz z wszystkimi elementami potrzebnymi (dataset, dataset_loader)

    U-Net/
        U-Net masks/ - maski wygenerowane dla danych treningowych w celu wykorzystania do uczenia modelu U-Net
        dataset.py - program zawierający definicję klasy dataset wraz z metodami
        import_csv_unet.py - program odczytujący dane wygenerowane podczas uczenia modelu i tworzący na ich podstawie wykresy których użyto w pracy
        train.py - program zawierający pętlę treningową U-Net oraz augmentację danych
        U-Net model code.py - program zawierający klasę U-Net wraz z metodami
        utils.py - program zawierający szereg funkcji importowanych w train.py w celu poprawy czytelności kodu

    Utils
        adjust_brightness.py - program stosujący metodę CLAHE dla wszystkich zdjęć w określonej lokalizacji i zapisujący powstałe zdjęcia w innej
        create_masks_from_coco.py - program odczytujący zawartość pliku results.json, na jego podstawie tworzący maski binarne dla każdego obrazu z katalogu images/
        image_plot.py - program zapisujący przykładowe zdjęcie w 6 zaugmentowanych wariantach
        process_csv_to_graph - odpowiednik programu import_csv_unet.py dla wyników modelu F-RCNN
        create_photo_stack.py - program przechodzący kolejno przez foldery zapisanych wersji modeli F-RCNN w celu utworzenia kompozycji porównawczych z efektów ich pracy
        stack_images_all.py - program przechodzący kolejno przez foldery zapisanych wersji modeli U-Net w celu utworzenia kompozycji porównawczych z efektów ich pracy
        

from detect_traffic_light import prepare_datasets, train_model, MODEL_PATH, CLASS_MAP_PATH

if __name__ == "__main__":
    train_ds, val_ds, class_names = prepare_datasets("dataset")
    model, history = train_model(train_ds, val_ds, class_names, epochs=20)  # поменяй числа как нужно
    print("Training finished. Model saved to:", MODEL_PATH)
    print("Class map saved to:", CLASS_MAP_PATH)

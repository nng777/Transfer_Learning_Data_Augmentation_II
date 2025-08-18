"""
1.Choose a Dataset:
1.1.Use a classic dataset like CIFAR-10 or CIFAR-100 (tensorflow.keras.datasets).

2.Build the Initial Mode:
2.1.Load a pre-trained model like MobileNetV2 as your base, without its final classification layer (include_top=False).
2.2.Freeze the weights of the base model (base.trainable = False).
2.3.Add your own custom classification "head" on top of the base model. This should include a GlobalAveragePooling2D layer and a Dense output layer with the correct number of neurons for your chosen dataset.
2.4.Compile the model with an Adam optimizer and an appropriate loss function (e.g., SparseCategoricalCrossentropy).

3.Initial Training:
3.1.Train the model for a few epochs. At this stage, you are only training the weights of your custom classification head.
3.2.Record the validation accuracy.

4.Fine-Tuning:
4.1.Unfreeze the base model (base.trainable = True).
4.2.Freeze the majority of the layers in the base model, leaving only the top layers (e.g., the last 20) trainable. This allows the model to adapt its more complex feature extractors to your specific dataset.
4.3.Crucially, re-compile the model with a very low learning rate (e.g., 1e-5). This prevents the pre-trained weights from being destroyed.

5.Continue Training:
5.1.Continue training the model for several more epochs.
5.2.Monitor the validation accuracy. Check and report the improvement over the initial training phase.


"""
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import os
import logging  

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(),
              logging.FileHandler('training.log') ]
)

logger = logging.getLogger(__name__)

class CaliforniaHousingModel(tf.keras.Model):
    def __init__ (self, units=256, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.norm_wide = tf.keras.layers.Normalization()
        self.norm_deep = tf.keras.layers.Normalization()
        self.dense1 = tf.keras.layers.Dense(units, activation=activation)
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(units, activation=activation)
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.output_layer = tf.keras.layers.Dense(1)
        
        
    def call(self, inputs):
        input_wide, input_deep = inputs
        norm_wide = self.norm_wide(input_wide)
        norm_deep = self.norm_deep(input_deep)
        hidden1 = self.dense1(norm_deep)
        hidden2 = self.dense2(hidden1)
        concat = tf.keras.layers.concatenate([norm_wide, hidden2])
        output = self.output_layer(concat)
        return output
    
    def adapt(self, data_wide, data_deep):
        self.norm_wide.adapt(data_wide)
        self.norm_deep.adapt(data_deep)
        
     
    @staticmethod
    def build_california_housing_model(units=256, activation='relu', learning_rate=0.001, optimizer=tf.keras.optimizers.Adam):
        model = CaliforniaHousingModel(units=units, activation=activation)
        model.compile(
            optimizer=optimizer(learning_rate=learning_rate),
            loss='mse',
            metrics=['RootMeanSquaredError']
        )
        return model

data = fetch_california_housing()
X, y = data.data, data.target
logger.info(f'Dataset shape: {X.shape}, Target shape: {y.shape}')

# Split dataset: 80% train, 10% valid, 10% test
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)



X_train_wide, X_train_deep = X_train[:, :3], X_train[:, 3:]
X_valid_wide, X_valid_deep = X_valid[:, :3], X_valid[:, 3:]
X_test_wide, X_test_deep = X_test[:, :3], X_test[:, 3:]
logger.info("Data successfully split into training, validation, and test sets.")


model = CaliforniaHousingModel.build_california_housing_model()
model.adapt(X_train_wide, X_train_deep) 
logger

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)


history = model.fit((X_train_wide, X_train_deep), y_train, 
                    epochs=20,
                    callbacks=[early_stopping, lr_scheduler],
                     validation_data=((X_valid_wide, X_valid_deep), y_valid),
                    batch_size=32)
logger.info("Model training complete.")


test_loss, test_rmse = model.evaluate((X_test_wide, X_test_deep), y_test)
logger.info(f'Test RMSE: {test_rmse}')
# print(f'Test RMSE: {test_rmse}')


model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'california_housing_model.keras')
model.save(model_path)
logger.info(f'Model successfully saved to {model_path}')









    
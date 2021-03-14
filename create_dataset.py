from data_processing.mozilla_common_voice import MozillaCommonVoiceDataset
from data_processing.urban_sound_8K import UrbanSound8K
from data_processing.dataset import Dataset
import warnings

warnings.filterwarnings(action='ignore')

mozilla_basepath = 'D:/downloads/fr1/mcv/fr~'
urbansound_basepath = 'D:/downloads/UrbanSound8K/UrbanSound8k'

mcv = MozillaCommonVoiceDataset(mozilla_basepath, val_dataset_size=900)
clean_train_filenames, clean_val_filenames = mcv.get_train_val_filenames()

us8K = UrbanSound8K(urbansound_basepath, val_dataset_size=900)
noise_train_filenames, noise_val_filenames = us8K.get_train_val_filenames()


print(mozilla_basepath)
print(urbansound_basepath)

windowLength = 256
config = {'windowLength': windowLength,
          'overlap': round(0.25 * windowLength),
          'fs': 16000,
          'audio_max_duration': 0.8}

if __name__=='__main__':

    val_dataset = Dataset(clean_val_filenames, noise_val_filenames, **config)
    val_dataset.create_tf_record(prefix='val', subset_size=300)

#    train_dataset = Dataset(clean_train_filenames, noise_train_filenames, **config)
#    train_dataset.create_tf_record(prefix='train', subset_size=300) 


    ## Create Test Set
#    clean_test_filenames = mcv.get_test_filenames()

#    noise_test_filenames = us8K.get_test_filenames()

#    test_dataset = Dataset(clean_test_filenames, noise_test_filenames, **config)
#    test_dataset.create_tf_record(prefix='test', subset_size=300, parallel=False)

#print("Val dataset")
#print(val_dataset)

#print("clean test filenames dataset")
#print(clean_test_filenames[0])

#print("noise test filenames dataset")
#print(noise_test_filenames[0])
#print("Test dataset")
#print(test_dataset)
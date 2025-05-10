import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataloader.DataloaderInterface import DataloaderInterface

class Dataloader(DataloaderInterface):

    def train_dataset(self, **kwargs):
        mean, std = kwargs['mean'], kwargs['std']
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        cifar100_training = torchvision.datasets.CIFAR100(root=self.data_path, train=True, download=True, transform=transform_train)
        return cifar100_training
    
    def val_dataset(self, **kwargs):
        return super().val_dataset(**kwargs)
    
    def test_dataset(self, **kwargs):
        mean, std = kwargs['mean'], kwargs['std']
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        cifar100_test = torchvision.datasets.CIFAR100(root=self.data_path, train=False, download=True, transform=transform_test)
        return cifar100_test

    def train_dataloader(self, **kwargs):
        """ return training dataloader
        Args:
            data_path: path to cifar100 training python dataset
            mean: mean of cifar100 training dataset
            std: std of cifar100 training dataset
            batch_size: dataloader batchsize
            num_workers: dataloader num_works
            shuffle: whether to shuffle
        Returns: train_data_loader:torch dataloader object
        """
        batch_size, num_workers, shuffle = kwargs['batch_size'], kwargs['num_workers'], kwargs['shuffle']
        cifar100_training = self.train_dataset(mean=kwargs['mean'], std=kwargs['std'])
        cifar100_training_loader = DataLoader(
            cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        return cifar100_training_loader
    
    def test_dataloader(self, **kwargs):
        """ return test dataloader
        Args:
            mean: mean of cifar100 test dataset
            std: std of cifar100 test dataset
            path: path to cifar100 test python dataset
            batch_size: dataloader batchsize
            num_workers: dataloader num_works
            shuffle: whether to shuffle
        Returns: cifar100_test_loader:torch dataloader object
        """
        batch_size, num_workers, shuffle = kwargs['batch_size'], kwargs['num_workers'], kwargs['shuffle']
        cifar100_test = self.test_dataset(mean=kwargs['mean'], std=kwargs['std'])
        cifar100_test_loader = DataLoader(
            cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        return cifar100_test_loader
    
    def val_dataloader(self, **kwargs):
        return super().val_dataloader(**kwargs)

# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader

# from interface.DataloaderInterface import DataloaderInterface

# class Dataloader(DataloaderInterface):

#     def train_dataset(self, **kwargs):
#         return super().train_dataset(**kwargs)
    
#     def val_dataset(self, **kwargs):
#         return super().val_dataset(**kwargs)
    
#     def test_dataset(self, **kwargs):
#         return super().test_dataset(**kwargs)
    
#     def train_dataloader(self, **kwargs):
#         """ return training dataloader
#         Args:
#             data_path: path to cifar100 training python dataset
#             mean: mean of cifar100 training dataset
#             std: std of cifar100 training dataset
#             batch_size: dataloader batchsize
#             num_workers: dataloader num_works
#             shuffle: whether to shuffle
#         Returns: train_data_loader:torch dataloader object
#         """
#         mean, std = kwargs['mean'], kwargs['std']
#         batch_size, num_workers, shuffle = kwargs['batch_size'], kwargs['num_workers'], kwargs['shuffle']

#         transform_train = transforms.Compose([
#             #transforms.ToPILImage(),
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomRotation(15),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])
#         #cifar100_training = CIFAR100Train(path, transform=transform_train)
#         # print(self.data_path)
#         # exit(0)
#         cifar100_training = torchvision.datasets.CIFAR100(root=self.data_path, train=True, download=True, transform=transform_train)
#         cifar100_training_loader = DataLoader(
#             cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

#         return cifar100_training_loader
    
#     def test_dataloader(self, **kwargs):
#         """ return training dataloader
#         Args:
#             mean: mean of cifar100 test dataset
#             std: std of cifar100 test dataset
#             path: path to cifar100 test python dataset
#             batch_size: dataloader batchsize
#             num_workers: dataloader num_works
#             shuffle: whether to shuffle
#         Returns: cifar100_test_loader:torch dataloader object
#         """
#         mean, std = kwargs['mean'], kwargs['std']
#         batch_size, num_workers, shuffle = kwargs['batch_size'], kwargs['num_workers'], kwargs['shuffle']

#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])
#         #cifar100_test = CIFAR100Test(path, transform=transform_test)
#         cifar100_test = torchvision.datasets.CIFAR100(root=self.data_path, train=False, download=True, transform=transform_test)
#         cifar100_test_loader = DataLoader(
#             cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

#         return cifar100_test_loader
    
#     def val_dataloader(self, **kwargs):
#         return super().val_dataloader(**kwargs)
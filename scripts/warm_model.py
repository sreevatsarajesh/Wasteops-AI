from wasteops.inference import service
if __name__=='__main__':
    service.load();print(f"Loaded {service.backend} on {service._device_value}")

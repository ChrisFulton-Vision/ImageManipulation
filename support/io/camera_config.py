from dataclasses import dataclass, field
import support.core.enums as enums
from support.io.my_logging import LOG

@dataclass
class CameraConfig:
    configFilepath: str = 'Configs/Default.yaml'
    saveFolder: str = ''
    calibFilepath: str = 'Calibrations/GenericAlvium864.txt'
    imageFilepath: str = None
    cam_index: int = 0

    image_processing_queue: list[dict] = field(default_factory=list)

    # numeric params
    secondsBetweenImages: float = 1.0
    aprilTagSize: float = 0.168
    cam_to_log_time_offset: float = 0.0
    yolo_conf: float = 0.75
    yolo_iou: float = 1.00
    target_fps: float = 20.0
    rt_speed: float = 1.0

    # sources
    imageSource: enums.ImageSource = None  # set default below in __post_init__
    ThreeDTruthFilepath: str = None
    yoloFilepath: str = ''
    hud_data_filepath: str = ''

    # export range
    export_quality: enums.ExportQuality = enums.ExportQuality.med_quality
    start_export_idx: int = 0
    end_export_idx: int = 1

    # playback / processing
    playback_mode: enums.PlaybackSpeed = None
    processingKernel: enums.ImageKernel = None

    # Vimba Usage
    use_vimba: bool = False
    vimba_profile: str = "Full Res"
    vimba_camera_id: str = ""
    vimba_settings_xml: str = ""

    # Vimba camera controls
    vimba_gain_auto: str = "Off"
    vimba_gain: float = 0.0
    vimba_exposure_auto: str = "Off"
    vimba_exposure_us: float = 10000.0

    # Data Processing tab defaults
    dp_img_dir: str = ''
    dp_conf_list: str = "0.80"
    dp_ckptN: int = 200
    dp_prefetch: int = 32

    def __post_init__(self):
        # Keep existing defaults if not provided
        if self.imageSource is None:
            self.imageSource = enums.ImageSource.Camera_Stream
        if self.playback_mode is None:
            self.playback_mode = enums.PlaybackSpeed.Fixed_fps
        if self.processingKernel is None:
            self.processingKernel = enums.ImageKernel.Unfiltered

    def copy(self, configToCopy):
        for obj in configToCopy.__dict__:
            try:
                self.__dict__[obj] = configToCopy.__dict__[obj]
            except KeyError as e:
                # Allows for versioning issues, changed naming conventions.
                LOG.info(f"Old cache loaded. Observe: {e}")
                pass

    @property
    def toDict(self):
        enum_classes = ['export_quality', 'imageSource', 'playback_mode', 'processingKernel']
        going_out = {}
        for attr in self.__dict__:
            if attr in ['configFilepath']:
                continue
            if attr in enum_classes:
                going_out[attr] = self.__getattribute__(attr).value
            else:
                going_out[attr] = self.__getattribute__(attr)
        return going_out

    def fromDict(self, my_dict: dict):
        enum_dict = {
            'export_quality': enums.ExportQuality,
            'imageSource': enums.ImageSource,
            'playback_mode': enums.PlaybackSpeed,
            'processingKernel': enums.ImageKernel
        }
        for key, value in my_dict.items():
            if hasattr(self, key):
                if key in enum_dict.keys():
                    self.__setattr__(key, enum_dict[key](value))
                else:
                    self.__setattr__(key, value)
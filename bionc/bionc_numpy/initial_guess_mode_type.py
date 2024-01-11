from enum import Enum


class InitialGuessModeType(Enum):
    FROM_CURRENT_MARKERS = ("FromCurrentMarkers",)
    USER_PROVIDED = ("UserProvided",)
    USER_PROVIDED_FIRST_FRAME_ONLY = ("UserProvidedFirstFrameOnly",)
    FROM_FIRST_FRAME_MARKERS = "FromFirstFrameMarkers"

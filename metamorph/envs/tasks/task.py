import copy
import os
from metamorph.config import cfg
from metamorph.envs.modules.agent import create_agent_xml
from metamorph.envs.tasks.escape_bowl import make_env_escape_bowl
from metamorph.envs.tasks.locomotion import make_env_locomotion
from metamorph.envs.tasks.obstacle import make_env_obstacle
from metamorph.envs.tasks.incline import make_env_incline
from metamorph.envs.tasks.exploration import make_env_exploration
from metamorph.envs.wrappers.select_keys import SelectKeysWrapper
from metamorph.utils import file as fu


def make_env(agent_name, kwargs={"corruption_level": 0}):
    xml_path = os.path.join(
        cfg.ENV.WALKER_DIR, "xml", "{}.xml".format(agent_name)
    )

    kwargs = copy.deepcopy(kwargs)
    kwargs["initial_xml_path"] = xml_path

    xml = create_agent_xml(xml_path)
    env_func = "make_env_{}".format(cfg.ENV.TASK)
    env = globals()[env_func](xml, agent_name, kwargs)

    if False:
        img = env.render(mode="rgb_array")
        show_rgb_image(img)

        env.reset()
        img = env.render(mode="rgb_array")
        show_rgb_image(img)

    # Add common wrappers in the end
    keys_to_keep = cfg.ENV.KEYS_TO_KEEP + cfg.MODEL.OBS_TYPES
    env = SelectKeysWrapper(env, keys_to_keep=keys_to_keep)
    return env




def show_rgb_image(rgb_array, window_name="Image"):
    import cv2
    import numpy as np
    """
    Displays an RGB image using OpenCV.

    Parameters:
        rgb_array (np.ndarray): Image in RGB format.
        window_name (str): Name of the OpenCV window.
    """
    # Convert RGB to BGR since OpenCV uses BGR format
    bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    # Show the image
    cv2.imshow(window_name, bgr_image)

    # Wait until a key is pressed, then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

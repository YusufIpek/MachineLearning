State: 

The state consists of the the position and velocity
of the car.

Action: 

One out of three possible actions can be taken:

push left = 0 , no push = 1, push right = 2

Reward: 

-1 for each time step, until the goal position of
0.5 is reached



about the gym core


    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.


def step(self, action):
    """Run one timestep of the environment's dynamics. When end of
    episode is reached, you are responsible for calling `reset()`
    to reset this environment's state.
    Accepts an action and returns a tuple (observation, reward, done, info).
    Args:
        action (object): an action provided by the environment
    Returns:
        observation (object): agent's observation of the current environment
        reward (float) : amount of reward returned after previous action
        done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
        info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
    """


def reset(self):
    """Resets the state of the environment and returns an initial observation.
    Returns: observation (object): the initial observation of the
        space.
    """

def render(self, mode='human'):
    """Renders the environment.
    The set of supported modes varies per environment. (And some
    environments do not support rendering at all.) By convention,
    if mode is:
    - human: render to the current display or terminal and
      return nothing. Usually for human consumption.
    - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
      representing RGB values for an x-by-y pixel image, suitable
      for turning into a video.
    - ansi: Return a string (str) or StringIO.StringIO containing a
      terminal-style text representation. The text can include newlines
      and ANSI escape sequences (e.g. for colors).
    Note:
        Make sure that your class's metadata 'render.modes' key includes
          the list of supported modes. It's recommended to call super()
          in implementations to use the functionality of this method.
    Args:
        mode (str): the mode to render with
        close (bool): close all open renderings
     """

 def seed(self, seed=None):
    """Sets the seed for this env's random number generator(s).
    Note:
        Some environments use multiple pseudorandom number generators.
        We want to capture all such seeds used in order to ensure that
        there aren't accidental correlations between multiple generators.
    Returns:
        list<bigint>: Returns the list of seeds used in this env's random
          number generators. The first value in the list should be the
          "main" seed, or the value which a reproducer should pass to
          'seed'. Often, the main seed equals the provided 'seed', but
          this won't be true if seed=None, for example.
    """
    logger.warn("Could not seed environment %s", self)
    return
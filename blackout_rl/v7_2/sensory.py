"""S0: deterministic egocentric rendering of legal semantic observations only."""
import math
import numpy as np

PALETTE = np.array([[0,0,0],[.35,.35,.35],[0,.5,1],[1,.2,.1],[.1,1,.2],[1,0,.5],
                    [1,1,.2],[0,1,1],[.5,0,1],[1,.5,0],[.5,.5,1]], dtype=np.float32)


class SemanticRetina:
    def __init__(self, width=128, height=64, fov_degrees=270, max_distance=48):
        self.width, self.height = width, height
        self.fov = math.radians(fov_degrees)
        self.max_distance = max_distance
        self.angles = np.linspace(-self.fov/2, self.fov/2, width)

    def render(self, observation, absolute_slot, heading):
        graphic = np.asarray(observation['graphic'], dtype=np.float32)
        if graphic.shape == (11,96,96):
            graphic = graphic.transpose(1,2,0)
        if graphic.shape != (96,96,11) or not np.isfinite(graphic).all():
            raise ValueError('expected finite 96×96×11 semantic observation')
        # Semantic image row zero is north; world Y increases northward.
        world = np.flip(graphic, axis=0)
        position = np.asarray(observation['vector'])[:90].reshape(10,9)[absolute_slot,:2] * 96
        distance = np.arange(1, self.max_distance+1, dtype=float)
        angles = heading + self.angles
        x = np.floor(position[0] + np.cos(angles[:,None])*distance).astype(int)
        y = np.floor(position[1] + np.sin(angles[:,None])*distance).astype(int)
        outside = (x<0)|(x>=96)|(y<0)|(y>=96)
        samples = world[np.clip(y,0,95),np.clip(x,0,95)].copy()
        samples[:,:,0] = 0
        samples[outside] = 0; samples[:,:,1][outside] = 1
        wall = samples[:,:,1] > .5
        visible = np.cumsum(wall, axis=1) <= 1
        # A wall itself is visible, everything after the first wall is occluded.
        visible &= np.concatenate((np.ones((self.width,1),bool), np.cumsum(wall,axis=1)[:,:-1] == 0),axis=1)
        active = (samples.max(axis=-1) > .5) & visible
        hit = active.argmax(axis=1)
        present = active.any(axis=1)
        channel = samples[np.arange(self.width), hit].argmax(axis=-1)
        colors = PALETTE[channel] / (1 + distance[hit,None] / 48)
        colors[~present] = 0
        # Fixed perspective-height proxy; no targets, rankings or planner overlays.
        extent = np.maximum(1, (self.height/(1+distance[hit]/3)).astype(int))
        rows = np.abs(np.arange(self.height)[:,None]-self.height/2) < extent[None,:]
        return (colors[None,:,:]*rows[:,:,None]).astype(np.float32)

    def sample(self, image, mapping):
        az = np.array([p['azimuth_degrees'] for p in mapping])
        el = np.array([p['elevation_degrees'] for p in mapping])
        x = np.clip(np.rint((np.radians(az)/self.fov+.5)*(self.width-1)).astype(int),0,self.width-1)
        y = np.clip(np.rint((.5-el/144)*(self.height-1)).astype(int),0,self.height-1)
        rgb = image[y,x]
        luminance = rgb @ np.array([.2126,.7152,.0722], dtype=np.float32)
        green = np.maximum(0, rgb[:,1]-.5*(rgb[:,0]+rgb[:,2]))
        return luminance + .25*green

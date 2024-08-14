# Imports and settings for plotting of graphs
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import imageio
import pathlib
import tqdm

pio.templates["custom"] = go.layout.Template(
    layout=go.Layout(
        margin=dict(l=20, r=20, t=40, b=0)
    )
)
pio.templates.default = "simple_white+custom"


class AnimationButtons():
    @staticmethod
    def play_scatter(frame_duration = 500, transition_duration = 300):
        return dict(label="Play", method="animate", args=
                    [None, {"frame": {"duration": frame_duration, "redraw": False},
                            "fromcurrent": True, "transition": {"duration": transition_duration, "easing": "quadratic-in-out"}}])
    
    @staticmethod
    def play(frame_duration = 1000, transition_duration = 0):
        return dict(label="Play", method="animate", args=
                    [None, {"frame": {"duration": frame_duration, "redraw": True},
                            "mode":"immediate",
                            "fromcurrent": True, "transition": {"duration": transition_duration, "easing": "linear"}}])
    
    @staticmethod
    def regular_speed(frame_duration = 1000, transition_duration = 0):
        return dict(label="1x Speed", method="animate", args=
                    [None, {"frame": {"duration": frame_duration, "redraw": True},
                            "fromcurrent": True, 
                            "transition": {"duration": transition_duration, "easing": "linear"}}])
    
    @staticmethod
    def speed_up(frame_duration = 500, transition_duration = 0):
        return dict(label="2x Speed", method="animate", args=
                    [None, {"frame": {"duration": frame_duration, "redraw": True},
                            "fromcurrent": True, 
                            "transition": {"duration": transition_duration, "easing": "linear"}}])
    @staticmethod
    def speed_up2(frame_duration = 250, transition_duration = 0):
        return dict(label="4x Speed", method="animate", args=
                    [None, {"frame": {"duration": frame_duration, "redraw": True},
                            "fromcurrent": True, 
                            "transition": {"duration": transition_duration, "easing": "linear"}}])
    
    @staticmethod
    def speed_up3(frame_duration = 100, transition_duration = 0):
        return dict(label="8x Speed", method="animate", args=
                    [None, {"frame": {"duration": frame_duration, "redraw": True},
                            "fromcurrent": True, 
                            "transition": {"duration": transition_duration, "easing": "linear"}}])
    
    @staticmethod
    def slow_down(frame_duration = 2000, transition_duration = 0):
        return dict(label="0.5x Speed", method="animate", args=
                    [None, {"frame": {"duration": frame_duration, "redraw": True},
                            "fromcurrent": True, 
                            "transition": {"duration": transition_duration, "easing": "linear"}}])
    
    @staticmethod
    def pause():
        return dict(label="Pause", method="animate", args=
                    [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
    
    @staticmethod
    def slider(frame_names):       
        steps= [dict(args=[[i], dict(frame={'duration': 300, 'redraw': False}, mode="immediate", transition= {'duration': 300})],
                           label=i+1, method="animate")
                for i, n in enumerate(frame_names)]
        
        return [dict(yanchor="top", xanchor="left",
                     currentvalue={'font': {'size': 16}, 'prefix': 'Frame: ', 'visible': True, 'xanchor': 'right'},
                     transition={'duration': 0, 'easing': 'linear'},
                     pad= {'b': 10, 't': 50},
                     len=0.9, x=0.1, y=0,
                     steps=steps)]


custom = [[0.0, "rgb(165,0,38)"],
          [0.1111111111111111, "rgb(215,48,39)"],
          [0.2222222222222222, "rgb(244,109,67)"],
          [0.3333333333333333, "rgb(253,174,97)"],
          [0.4444444444444444, "rgb(254,224,144)"],
          [0.5555555555555556, "rgb(224,243,248)"],
          [0.6666666666666666, "rgb(171,217,233)"],
          [0.7777777777777778, "rgb(116,173,209)"],
          [0.8888888888888888, "rgb(69,117,180)"],
          [1.0, "rgb(49,54,149)"]]

def frames_to_gif(frames, filename) -> None:
    print(f"Creating gif: {filename}")
    images = []
    for i, frame in tqdm.tqdm(enumerate(frames)):
        fig = go.Figure(data=frame["data"], layout=frame["layout"])
        name = f"{filename.split('.')[0]}_{i}.png"
        fig.write_image(name, format='png', engine='kaleido')
        images.append(imageio.v2.imread(name))
        pathlib.Path(name).unlink()
    imageio.mimsave(filename, images, duration=2)

   
def animation_to_gif(fig, filename, frame_duration=100, width=1200, height=800):
    import gif
    @gif.frame
    def plot(f, i):
        f_ = go.Figure(data=f["frames"][i]["data"], layout=f["layout"])
        f_["layout"]["updatemenus"] = []
        f_.update_layout(title=f["frames"][i]["layout"]["title"], width=width, height=height)
        return f_

    gif.save([plot(fig, i) for i in range(len(fig["frames"]))], filename, duration=frame_duration)
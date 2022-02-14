"""COLOR-BLIND PALETTE, via https://www.nature.com/articles/nmeth.1618."""
from matplotlib.colors import rgb2hex


ORANGE = (230, 159, 0)  # cited
BLUE = (86, 180, 233)  # abstract
YELLOW = (240, 228, 66)  # title
GREEN = (0, 158, 115)  # A&T (intermediate)
DARK_BLUE = (0, 114, 178)  # impact

color_dict = {
    'impact': DARK_BLUE,
    'cited': ORANGE,
    'abstract_only': BLUE,
    'title_only': YELLOW,
    'both': GREEN,
}


def get_color_hex(rgb, alpha=1):
    rgba = [i / 255 for i in rgb] + [alpha]
    return rgb2hex(rgba)


CATEG_HEX = {i: get_color_hex(color_dict[i]) for i in color_dict}
CATEG_HEX.update({'title': CATEG_HEX['title_only'],
                  'abstract': CATEG_HEX['abstract_only']})

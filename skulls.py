import os
import requests, json

import numpy as np
import pandas as pd

from PIL import Image, ImageChops, ImageStat, ImageDraw, ImageFont
import qrcode

from matplotlib.pyplot import imshow

from itertools import product, combinations



# CONSTANTS ########################################################################################



special_tokens = [9, 19, 20, 24, 27, 36, 41, 42, 43, 70]



beard_groups = [
        [7, 0, 1, 2], # thin
        [4, 5, 6, 3], # thick
        [8], # NONE
    ]

eyes_groups = [
        [11, 17], # HARDCODE THESE: squint (no color), small
        [14, 15, 16], # alien, alien, alien,
        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], # side look
        [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 12], # small
        [13, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], # wide
        [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58], # sunk
        [59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73], # tall
        [74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88], # extra sunk
    ]

hair_groups = [
        [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61], # horns
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], # triple (horns + mohawk)
        [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], # short mohawk
        [32], # very deep mohawk
        [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], # deep mohawk
        [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76], # deep widows peak
        [77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92], # shallow widows peak
        [93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107], # emo
        [108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121], # side comb
        [122], # NONE
    ]

d_flipped_nose_map = {4:2, 5:3, 7:6}



# GETTING COLORS FOR PROPERTIES ####################################################################


def rgb_to_hex(rgb):
    if type(rgb) != tuple:
        return rgb
    return '#%02x%02x%02x' % rgb # rgb is a tuple of 3 0-255 int



def hex_to_rgb(hex):
    if type(hex) != str:
        return hex
    return tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))



def get_background_color_map():
    d_background_color = {}
    for background_id in range(132):
        token_ids = df_meta[df_meta['backgroundId'] == background_id]['id'].values
        token_ids = list(set(token_ids) - set(special_tokens))
        color = cropped_skulls[token_ids[0]].load()[0, 0]
        if color != cropped_skulls[token_ids[1]].load()[0, 0]:
            raise
        d_background_color.update({background_id:color})
    return d_background_color



def get_skull_color_map():
    d_skull_color = {}
    for skull_gene in range(15):
        token_ids = df_meta[df_meta['skullGene'] == skull_gene]['id'].values
        token_ids = list(set(token_ids) - set(special_tokens))
        color = cropped_skulls[token_ids[0]].load()[11, 19]
        if color != cropped_skulls[token_ids[1]].load()[11, 19]:
            raise
        d_skull_color.update({skull_gene:color})
    return d_skull_color



def get_bones_color_map():
    d_bones_color = {}
    for bones_gene in range(15):
        token_ids = df_meta[df_meta['bonesGene'] == bones_gene]['id'].values
        token_ids = list(set(token_ids) - set(special_tokens))
        color = cropped_skulls[token_ids[0]].load()[4, 20]
        if color != cropped_skulls[token_ids[1]].load()[4, 20]:
            raise
        d_bones_color.update({bones_gene:color})
    return d_bones_color



def get_beard_color_map():
    d_beard_color = {}
    for i, beard_group in enumerate(beard_groups):
        if beard_group == beard_groups[-1]:
            continue
        for beard_gene in beard_group:
            token_ids = df_meta[df_meta['beardGene'] == beard_gene]['id'].values
            token_ids = list(set(token_ids) - set(special_tokens))
            color = cropped_skulls[token_ids[0]].load()[11, 23]
            if color != cropped_skulls[token_ids[1]].load()[11, 23]:
                raise
            d_beard_color.update({beard_gene:color})
    return d_beard_color



def get_eyes_color_map():
    d_eyes_color = {}
    for i, eyes_group in enumerate(eyes_groups):
        if i in [0, 1]:
            continue
        y = 9 if i == 2 else 10
        for eyes_gene in eyes_group:
            token_ids = df_meta[df_meta['eyesGene'] == eyes_gene]['id'].values
            token_ids = list(set(token_ids) - set(special_tokens))
            color = cropped_skulls[token_ids[0]].load()[9, y]
            if color != cropped_skulls[token_ids[1]].load()[9, y]:
                raise
            d_eyes_color.update({eyes_gene:color})

    # alien eyes with pupils in group 1
    d_eyes_color[15] = (255, 255, 255)
    d_eyes_color[16] = (255, 255, 255)

    return d_eyes_color



def get_hair_color_map():
    d_hair_color = {}
    for i, hair_group in enumerate(hair_groups):
        if hair_group == hair_groups[-1]:
            continue
        x = 4 if i == 0 else 11
        for hair_gene in hair_group:
            token_ids = df_meta[df_meta['hairGene'] == hair_gene]['id'].values
            token_ids = list(set(token_ids) - set(special_tokens))
            color = cropped_skulls[token_ids[0]].load()[x, 3]
            if color != cropped_skulls[token_ids[1]].load()[x, 3]:
                raise
            d_hair_color.update({hair_gene:color})
    return d_hair_color



# MAPPING GENES TO GENE GROUPS #####################################################################



def get_group(gene_int, groups):
    for i, group in enumerate(groups):
        if gene_int in group:
            return i
    raise



def get_beard_group(gene_int):
    return get_group(gene_int, beard_groups)

def get_eyes_group(gene_int):
    return get_group(gene_int, eyes_groups)

def get_hair_group(gene_int):
    return get_group(gene_int, hair_groups)



# RE-COLORING HELPER ###############################################################################



def replace_black(im, color):
    #https://stackoverflow.com/a/3753428

    data = np.array(im)   # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T # Temporarily unpack the bands for readability

    # Replace black with color... (leaves alpha values alone...)
    black_areas = (red == 0) & (blue == 0) & (green == 0)
    data[..., :-1][black_areas.T] = color # Transpose back needed

    return Image.fromarray(data)



# INTERACTING WITH ONLINE METADATA #################################################################



def get_raw_token_metadata_old(token_id):

    url = 'https://cryptoskulls.com/api/token/%d' % token_id
    return json.loads(requests.get(url).text)



def get_metadata_old(full_refresh=False):

    if full_refresh:
        l_d = []
        for token_id in range(10000):
            response = get_raw_token_metadata_old(token_id)
            try:
                d = {att.get('trait_type', 'feature'):att['value'] for att in response['attributes']}
                d.update({'token_id': token_id})
                l_d.append(d.copy())
            except:
                print(response)
        df_meta = pd.DataFrame.from_dict(l_d)
        df_meta.to_csv('metadata_old.csv', index=False)
        return df_meta
    else:
        return pd.read_csv('metadata_old.csv')



def get_raw_token_metadata(token_id):

    url = 'https://goldofskulls.com/api/tokens/%d' % token_id
    return json.loads(requests.get(url).text)



def get_metadata(full_refresh=False):

    if full_refresh:
        l_d = []
        for token_id in range(10000):
            response = get_raw_token_metadata(token_id)[0]

            try:
                d = response.copy()
                d.pop('imageUrl', None)
                d.pop('traits', None)
                d.pop('skillsString', None)
                l_d.append(d.copy())
            except Exception as e:
                print(response)
                raise e

        df_meta = pd.DataFrame.from_dict(l_d)
        df_meta.to_csv('metadata.csv', index=False)
        return df_meta
    else:
        df_meta = pd.read_csv('metadata.csv')
        df_meta.loc[9999, 'backgroundId'] = 131
        df_meta.loc[9999, 'skullGene'] = 0
        df_meta.loc[9999, 'hairGene'] = 89
        df_meta.loc[9999, 'eyesGene'] = 47
        df_meta.loc[9999, 'noseGene'] = 6
        df_meta.loc[9999, 'teethGene'] = 10
        df_meta.loc[9999, 'bonesGene'] = 10
        df_meta.loc[9999, 'beardGene'] = 8
        return df_meta


# EXTRACTING 24x24 IMAGES FROM MAIN SKULL IMAGE ####################################################



def get_boxes(img, box_size=24):
    
    lx = range(0, img.size[0] + box_size, box_size)
    ly = range(0, img.size[1] + box_size, box_size)
    
    return [(tx[0], ty[0], tx[1], ty[1]) for ty, tx in product(zip(ly[:-1], ly[1:]), zip(lx[:-1], lx[1:]), repeat=True)]



def get_skull_images():

    d = 24
    im_skulls = Image.open('cryptoskulls.png').convert('RGB')#.crop((0, 0, d*n, d*n))
    skull_boxes = get_boxes(im_skulls, d)
    cropped_skulls = [im_skulls.crop(skull_box) for skull_box in skull_boxes]
    return cropped_skulls



# CHEATY PRE-CALCULATIONS ##########################################################################



df_meta = get_metadata()
cropped_skulls = get_skull_images()

d_background_color = get_background_color_map()
d_skull_color = get_skull_color_map()
d_bones_color = get_bones_color_map()
d_beard_color = get_beard_color_map()
d_eyes_color = get_eyes_color_map()
d_hair_color = get_hair_color_map()



# RARITY SCORES ####################################################################################



base_traits = ['backgroundId', 'skullGene', 'hairGroup', 'eyesGroup',
               'noseGene', 'teethGene', 'bonesGene', 'beardGene']

skill_traits = ['fireLord', 'regenerationLord', 'iceLord', 'poisonLord',
                'lightningLord', 'archmage', 'immortal', 'fastKick', 'ironFists']
game_traits = ['intelligence', 'strength', 'dexterity', 'stamina', 'accuracy', 'speed']

hr_colors = ['backgroundColor', 'skullColor', 'bonesColor', 'beardColor', 'eyesColor', 'hairColor']

score_cols = ['base_score', 'skill_score', 'game_score', 'hidden_score']
score_cols_s = [score_col + '_s' for score_col in score_cols]


def count_unique_colors(dr):
    colors = dr[hr_colors].unique().tolist()
    colors = [color for color in colors if color not in ['#263238', np.nan]]
    return len(colors)


def calc_rarity(df_meta):

    df_meta = df_meta.copy() # starting point

    # PRE CALCULATIONS

    # base traits
    df_meta['hairGroup'] = df_meta['hairGene'].map(get_hair_group)
    df_meta['eyesGroup'] = df_meta['eyesGene'].map(get_eyes_group)
    df_meta['beardGroup'] = df_meta['beardGene'].map(get_beard_group)

    df_meta['backgroundColor'] = df_meta['backgroundId'].map(d_background_color).map(rgb_to_hex)
    df_meta['skullColor'] = df_meta['skullGene'].map(d_skull_color).map(rgb_to_hex)
    df_meta['bonesColor'] = df_meta['bonesGene'].map(d_bones_color).map(rgb_to_hex)
    df_meta['beardColor'] = df_meta['beardGene'].map(d_beard_color).map(rgb_to_hex)
    df_meta['eyesColor'] = df_meta['eyesGene'].map(d_eyes_color).map(rgb_to_hex)
    df_meta['hairColor'] = df_meta['hairGene'].map(d_hair_color).map(rgb_to_hex)

    df_meta['base_count'] = 8 - df_meta[['hairColor', 'bonesColor', 'beardColor']].isna().sum(axis=1)
    df_meta['color_count'] = df_meta.apply(count_unique_colors, axis=1)

    # skill traits
    df_meta['fastKick'] = df_meta['skills'].str.contains('FAST_KICK')
    df_meta['ironFists'] = df_meta['skills'].str.contains('IRON_FISTS')
    
    df_meta['skill_count'] = df_meta[skill_traits].sum(axis=1)

    # SCORE CALCULATIONS

    # base traits score
    # reference: https://datascienceparichay.com/article/jaccard-similarity-python/
    if False:
        # warning, time-consuming computation
        jaccard = lambda a, b: 1 - np.sum(a == b) / 8
        df_pleb = df_meta[df_meta['uniquenessIndex'] > 1].copy()
        df_meta.loc[df_pleb.index, 'base_score'] = df_pleb[base_traits].T.corr(method=jaccard).mean()
        # df_meta[['id', 'base_score_j']].to_csv('jaccard.csv', index=False)
    else:
        df_jaccard = pd.read_csv('jaccard.csv').rename(columns={'jaccard': 'base_score'})
        df_meta = df_meta.merge(df_jaccard, 'left', on='id')
    
    df_meta.loc[df_meta['uniquenessIndex'] == 1, 'base_score'] = 1 # lords get max distance
        
    # skill traits score
    ds_skill_score = (df_meta[skill_traits].replace({False:np.nan}) * 10000 / df_meta[skill_traits].sum()).product(axis=1)
    df_meta['skill_score'] = np.log(ds_skill_score)
    # df_meta['skill_score'] = (10000 * df_meta[skill_traits] / df_meta[skill_traits].sum()).sum(axis=1)

    # skill combos (redundant with skill score)
    # df_meta['combo_score'] = 10000 / df_meta['skills'].map(df_meta.groupby('skills').apply(len))

    # game traits
    df_meta['game_score'] = df_meta[game_traits].product(axis=1)
    # df_meta['game_score'] = (df_meta[game_traits] / df_meta[game_traits].mean()).sum(axis=1)

    # hidden rarities

    df_meta['match_code'] = 0
    df_meta['match_desc'] = ''

    for col1, col2, col3 in combinations(hr_colors, 3):
        i_match = (df_meta[col1] == df_meta[col2]) & (df_meta[col1] == df_meta[col3]) & ~df_meta[col1].isna()
        if i_match.any():
            df_meta.loc[i_match, 'match_code'] += 10
            df_meta.loc[i_match, 'match_desc'] += '%s=%s=%s' % (col1[:-5], col2[:-5], col3[:-5])# + '-' + df_meta.loc[i_match, col1]
            
    for col1, col2 in combinations(hr_colors, 2):
        i_match = (df_meta[col1] == df_meta[col2]) & ~df_meta[col1].isna() & ~df_meta['match_desc'].str.contains(col1[:-5])
        if i_match.any():
            df_meta.loc[i_match, 'match_code'] += 1
            str_spacer = np.where(df_meta.loc[i_match, 'match_desc'] == '', '', ' & ')
            df_meta.loc[i_match, 'match_desc'] = df_meta.loc[i_match, 'match_desc'] + str_spacer + '%s=%s' % (col1[:-5], col2[:-5])# + '-' + df_meta.loc[i_match, col1]

    # lords cannot have color matching (even if their original metadata says so)
    df_meta.loc[df_meta['uniquenessIndex'] == 1, 'match_code'] = 1337
    df_meta.loc[df_meta['uniquenessIndex'] == 1, 'match_desc'] = ''

    df_meta['hidden_score'] = 10000 / df_meta['match_code'].map(df_meta.groupby('match_code').apply(len))

    # scaling
    for score_col, score_col_s in zip(score_cols, score_cols_s):
        v_min = df_meta[score_col].min()
        v_max = df_meta[score_col].max()
        df_meta[score_col_s] = (df_meta[score_col] - v_min) / (v_max - v_min)

    df_meta['rarity_score'] = df_meta[score_cols_s].sum(axis=1)
    df_meta['rarity_score_r'] = df_meta['rarity_score'].rank(ascending=False)

    return df_meta



# COLOR DISTANCE STUFF #############################################################################



def color_distance_a(value1, value2):
    return np.linalg.norm(np.array(hex_to_rgb(value1)) - np.array(hex_to_rgb(value2)))

def color_distance(df, col1, col2):
    return df.apply(lambda dr: color_distance_a(dr[col1], dr[col2]), axis=1)



# COMPOSITING ######################################################################################



def project_compare(project_a, project_b, resize=96, token_ids=None):

    if token_ids == None:
        token_ids = [7583] + get_comparison_ids() + [6969]
        print(token_ids)

    ims_a = [assemble(token_id, project_a, resize=resize) for token_id in token_ids]
    ims_b = [assemble(token_id, project_b, resize=resize) for token_id in token_ids]

    n_wide = int(np.floor(np.sqrt(len(token_ids) / 2)))
    n_wide = max(n_wide, 1)

    im_out = Image.new(mode='RGBA', size=(resize*2*n_wide, resize*int(np.ceil(len(token_ids)/n_wide))))

    boxes = get_boxes(im_out, resize)

    for i, (im_a, im_b) in enumerate(zip(ims_a, ims_b)):

        im_out.paste(im_a, boxes[2*i])
        im_out.paste(im_b, boxes[2*i + 1])


        # im_out.paste((im_a if i % 2 == 0 else im_b)[i // 2], box)

    return im_out



def get_comparison_ids():

    df = df_meta.copy()

    token_ids = []

    df['beardGroup'] = df['beardGene'].map(get_beard_group)
    df['eyesGroup'] = df['eyesGene'].map(get_eyes_group)
    df['hairGroup'] = df['hairGene'].map(get_hair_group)

    token_ids += df.groupby('beardGroup').apply(lambda df: df.index[-1]).tolist()
    token_ids += df.groupby('eyesGroup').apply(lambda df: df.index[-1]).tolist()
    token_ids += df.groupby('noseGene').apply(lambda df: df.index[-1]).tolist()
    token_ids += df.groupby('teethGene').apply(lambda df: df.index[-1]).tolist()
    token_ids += df.groupby('hairGroup').apply(lambda df: df.index[-1]).tolist()

    return sorted(list(set(token_ids)))



# ASSEMBLE! ########################################################################################



d_project = {
            'cryptoskulls_backup':      {'dim': 24, 'flip_noses': True},
            'cryptoskulls_24':          {'dim': 24, 'flip_noses': True},
            'cryptoskulls_24_profile':  {'dim': 24, 'flip_noses': False},
            'cryptoskulls_96':          {'dim': 96, 'flip_noses': True},
            }



def assemble(token_id, project_name, resize=None, bg_visible=True):

    dim = d_project[project_name]['dim']

    d_meta = df_meta[df_meta['id'] == token_id].iloc[0].to_dict()
    # background
    color_background = d_background_color[d_meta['backgroundId']] if bg_visible else None
    im_out = Image.new(mode='RGBA', size=(dim, dim), color=color_background)
    # bones
    if d_meta['bonesGene'] != 15:
        color_bones = d_bones_color[d_meta['bonesGene']]
        im_bones = Image.open('%s\%s\%s' % (project_name, 'bones', 'bones0.png'))
        im_out.paste(replace_black(im_bones, color_bones), im_bones)
    # skull
    color_skull = d_skull_color[d_meta['skullGene']]
    im_skull = Image.open('%s\%s\%s' % (project_name, 'skull', 'skull0.png'))
    im_out.paste(replace_black(im_skull, color_skull), im_skull)
    # beard
    if d_meta['beardGene'] != 8:
        type_beard = get_beard_group(d_meta['beardGene'])
        color_beard = d_beard_color[d_meta['beardGene']]
        im_beard = Image.open('%s\%s\%s' % (project_name, 'beard', 'beard%d.png' % type_beard))
        im_out.paste(replace_black(im_beard, color_beard), im_beard)
    # eyes
    type_eyes = get_eyes_group(d_meta['eyesGene'])
    if type_eyes in [0, 1]:
        # hard coded
        im_eyes = Image.open('%s\%s\%s' % (project_name, 'eyes', '#%s.png' % d_meta['eyesGene']))
        im_out.paste(im_eyes, im_eyes)
    else:
        color_eyes = d_eyes_color[d_meta['eyesGene']]
        im_eyes = Image.open('%s\%s\%s' % (project_name, 'eyes', 'eyes%d.png' % type_eyes))
        im_out.paste(replace_black(im_eyes, color_eyes), im_eyes)
    # nose
    nose_gene = d_meta['noseGene']
    nose_gene_use = d_flipped_nose_map.get(nose_gene, nose_gene)
    im_nose = Image.open('%s\%s\%s' % (project_name, 'nose', '#%s.png' % nose_gene_use))
    if nose_gene in d_flipped_nose_map.keys() and d_project[project_name]['flip_noses']:
        im_nose = im_nose.transpose(Image.FLIP_LEFT_RIGHT)
    im_out.paste(im_nose, im_nose)
    # teeth
    im_teeth = Image.open('%s\%s\%s' % (project_name, 'teeth', '#%s.png' % d_meta['teethGene']))
    im_out.paste(im_teeth, im_teeth)
    # hair
    if d_meta['hairGene'] != 122:
        type_hair = get_hair_group(d_meta['hairGene'])
        color_hair = d_hair_color[d_meta['hairGene']]
        if type_hair == 1:
            im_hair = Image.open('%s\%s\%s' % (project_name, 'hair', 'hair%d.png' % 0))
            im_hair3 = Image.open('%s\%s\%s' % (project_name, 'hair', 'hair%d.png' % 2))
            im_hair.paste(im_hair3, im_hair3)
        else:
            im_hair = Image.open('%s\%s\%s' % (project_name, 'hair', 'hair%d.png' % type_hair))
        im_out.paste(replace_black(im_hair, color_hair), im_hair)

    if resize is not None:
        im_out = im_out.resize((resize, resize), Image.NEAREST)

    return im_out



# EVALUATING ASSEMBLY RESULTS ######################################################################



def calc_diff(im1, im2):
    return sum(ImageStat.Stat(ImageChops.difference(im1, im2)).rms)



def validate():

    for token_id in range(1, 10000):
        if token_id in special_tokens:
            continue
        im_new = assemble(token_id, 'cryptoskulls_backup').convert('RGB')
        im_old = cropped_skulls[token_id]
        diff = calc_diff(im_new, im_old)
        if diff > 0:
            print(token_id)
            print(diff)
            imshow(im_new)
            raise
    print('success')



def export_project(project_name, filepath_template, resize=None):

    for token_id in range(0, 10000):
        if token_id % 1000 == 0:
            print('starting ', token_id)
        if token_id in special_tokens:
            continue
        im = assemble(token_id, project_name)
        if resize is not None:
            im = im.resize((resize, resize), Image.NEAREST)
        im.save(filepath_template % token_id)



# SVG CONVERSION - NAIVE ###########################################################################
# kinda-works but edges are shared which leads to artifacts



def export_svgs(filepath_template, bg_visible=True):
    for token_id in range(0, 10000):
        if token_id in special_tokens:
            continue
        im = cropped_skulls[token_id]
        produce_svg(im, token_id, filepath_template, bg_visible)



def svg_polygon(points, rgb_color):
    str_points = ' '.join(['%d,%d' % (x, y) for x, y in points])
    return """<polygon points="%s" fill="rgb(%d,%d,%d)"/>""" % (str_points, *rgb_color)



c_outline = (38, 50, 56)
c_trans = (0,0,0,0)
pixels_skull = Image.open('%s\%s\%s' % ('cryptoskulls_backup', 'skull', 'skull0.png')).load()

# sophisticated
def produce_svg(im, token_id, filepath_template, bg_visible=True):

    line = '' #'\n'

    # im = cropped_skulls[token_id]
    pixels = im.copy().load()
    pixels_alt = im.copy().load()
    
    # background
    bg_pixel = pixels[0, 0]
    svg_content = ''
    if bg_visible:
        svg_content += """<rect width="24" height="24" style="fill:rgb(%d,%d,%d)"/>""" % bg_pixel

    # outline
    pt_alt = None

    p_assigned = []

    for i, j in product(range(24), range(24)):
        if (pixels[i, j] != bg_pixel) or (pixels_skull[i, j] != c_trans):
            pixels_alt[i, j] = c_outline
            if pt_alt is None:
                pt_alt = (i, j) # takes the first
        else:
            p_assigned += [(i, j)]

    svg_poly, p_polygon = get_polygon(pixels_alt, *pt_alt)
    svg_content += line + svg_poly

    # rest of the stuff
    
    for i, j in product(range(24), range(24)):
        if (pixels[i, j] != c_outline) and ((i, j) not in p_assigned):
            svg_poly, p_polygon = get_polygon(pixels, i, j)
            svg_content += line + svg_poly
            p_assigned += p_polygon

    svg = """<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24">%s</svg>""" % svg_content
    with open(filepath_template % token_id, 'w') as f:
        f.write(svg)



def get_polygon(pixels, i, j):
    rgb_color = pixels[i, j]
    points, p_polygon, p_processed = get_points(pixels, i, j, [], 'r')
    points = reduce_points(points)
    return svg_polygon(points, rgb_color), p_polygon



def get_points(pixels, i, j, p_processed, call):

    p_processed += [(i, j)]
    rgb_color = pixels[i, j]
    a, b, c, d = (i, j), (i+1, j), (i+1, j+1), (i, j+1)
    pt_u, pt_r, pt_d, pt_l = (i, j-1), b, d, (i-1, j)

    if j > 0 and pt_u not in p_processed and pixels[pt_u[0], pt_u[1]] == rgb_color:
        pts_u, pixels_u, p_processed = get_points(pixels, *pt_u, p_processed, 'u')
    else:
        pts_u, pixels_u = [], []

    if i < 23 and pt_r not in p_processed and pixels[pt_r[0], pt_r[1]] == rgb_color:
        pts_r, pixels_r, p_processed = get_points(pixels, *pt_r, p_processed, 'r')
    else:
        pts_r, pixels_r = [], []

    if j < 23 and pt_d not in p_processed and pixels[pt_d[0], pt_d[1]] == rgb_color:
        pts_d, pixels_d, p_processed = get_points(pixels, *pt_d, p_processed, 'd')
    else:
        pts_d, pixels_d = [], []

    if i > 0 and pt_l not in p_processed and pixels[pt_l[0], pt_l[1]] == rgb_color:
        pts_l, pixels_l, p_processed = get_points(pixels, *pt_l, p_processed, 'l')
    else:
        pts_l, pixels_l = [], []

    p_polygon = [(i, j)] + pixels_u + pixels_r + pixels_d + pixels_l
    if call == 'u':
        points = [d, *pts_l, a, *pts_u, b, *pts_r, c, *pts_d, d]
    elif call == 'r':
        points = [a, *pts_u, b, *pts_r, c, *pts_d, d, *pts_l, a]
    elif call == 'd':
        points = [b, *pts_r, c, *pts_d, d, *pts_l, a, *pts_u, b]
    elif call == 'l':
        points = [c, *pts_d, d, *pts_l, a, *pts_u, b, *pts_r, c]
    else:
        raise

    return points, p_polygon, p_processed



def reduce_points(points):
    # repeats
    count = 1
    while count > 0:
        count = 0
        n = len(points)
        for i in range(n)[::-1]:
            if i > 0 and points[i] == points[i-1]:
                points.pop(i)
                count += 1
    # tracebacks
    count = 1
    while count > 0:
        count = 0
        n = len(points)
        for i in range(n)[::-1]:
            if i > 0 and i < n-1 and points[i-1] == points[i+1]:
                points.pop(i+1)
                points.pop(i)
                count += 1
    # straight lines
    count = 1
    while count > 0:
        count = 0
        n = len(points)
        for i in range(n)[::-1]:
            if i > 0 and i < n-1 and (points[i-1][0] == points[i+1][0] or points[i-1][1] == points[i+1][1]):
                points.pop(i)
                count += 1
    if points[0] == points[-1]:
        points.pop(-1)
    return points



# CLONING PROJECT BASE IMAGES WITH RESIZE OPTION ###################################################



def clone_project(project_from, project_to, dimensions):

    # Example: skulls.clone_project('cryptoskulls_24', 'cryptoskulls_96', 96)

    for trait in os.listdir(project_from):
        dir_from = os.path.join(project_from, trait)
        dir_to = os.path.join(project_to, trait)
        if not os.path.exists(dir_to):
            os.makedirs(dir_to)
        for im_filename in os.listdir(dir_from):
            im = Image.open(os.path.join(dir_from, im_filename))
            im = im.resize((dimensions, dimensions), Image.NEAREST)
            im.save(os.path.join(dir_to, im_filename))



# SKULL MOSAICS ####################################################################################



def create_skull_mosaic(skull_id, *args, **kwargs):

    # handle lists of skulls
    if type(skull_id) == list:
        filenames = [create_skull_mosaic(skull_id_i, *args, **kwargs) for skull_id_i in skull_id]
        return filenames

    im_raw = cropped_skulls[skull_id]
    im_scaled = im_raw.copy().resize((im_raw.size[0]*4, im_raw.size[1]*4), Image.NEAREST)

    im_padded = Image.new('RGB', (100, 100), im_scaled.load()[0, 0])
    im_padded.paste(im_scaled, (2, 2))

    im_og = im_padded.copy()

    output_file_base_name = 'cryptoskull mosaic #%d' % skull_id

    return create_mosaic(im_og, output_file_base_name, *args, **kwargs)



def create_file_mosaic(filename, *args, **kwargs):

    im_og = Image.open(filename).convert('RGB')

    output_file_base_name = '%s mosaic' % filename.split('.')[0]

    return create_mosaic(im_og, output_file_base_name, *args, **kwargs)



def create_mosaic(im_og, output_file_base_name, gif_mode=False, verbose=False):

    d = 24

    # getting background colors which are used to decide what skulls to use
    skull_bg_pixels = np.array([cropped_skull.load()[0, 0] for cropped_skull in cropped_skulls])

    # breaking down into pixels, counts and prioritizing

    pixel_access = im_og.load()
    og_pixels = np.array([pixel_access[x, y] for y in range(im_og.size[1]) for x in range(im_og.size[0])])

    u_og_pixels = np.unique(og_pixels.copy(), axis=0)

    counts = [np.equal(og_pixels, u_og_pixel).all(axis=1).sum() for u_og_pixel in u_og_pixels]

    counts_sort_index = np.argsort(counts)
    u_og_pixels = u_og_pixels[counts_sort_index]
    counts = [counts[i] for i in counts_sort_index]

    if verbose:
        print(u_og_pixels)
        print(counts)

    # assembling
    im_out_blank = Image.new(mode='RGBA', size=(im_og.size[0] * d, im_og.size[1] * d))
    boxes_out = get_boxes(im_out_blank, d)
    
    n = 5 if gif_mode else 1
    l_im_out = [im_out_blank.copy() for i in range(n)]
    
    np.random.seed(500)

    used_rep_box_ids = np.array([])
    for u_og_pixel, count in zip(u_og_pixels, counts):
        dists = np.linalg.norm(u_og_pixel - skull_bg_pixels, axis=1) # distance
        sorted_box_ids = np.argsort(dists)
        sorted_box_ids = sorted_box_ids[~np.isin(sorted_box_ids, used_rep_box_ids)]
        rep_box_ids = sorted_box_ids[:count]
        used_rep_box_ids = np.append(used_rep_box_ids, rep_box_ids.copy())

        for im_out in l_im_out:
        
            np.random.shuffle(rep_box_ids)

            rep_box_id_id = 0
            for og_pixel, box_out in zip(og_pixels, boxes_out):
                if np.equal(og_pixel, u_og_pixel).all():
                    im_out.paste(cropped_skulls[rep_box_ids[rep_box_id_id]], box_out)
                    rep_box_id_id += 1

    # saving            
    if gif_mode:
        filename = '%s.gif' % output_file_base_name
        l_im_out[0].save(filename, save_all=True, append_images=l_im_out[1:], optimize=False, duration=250, loop=0)
    else:
        filename = '%s.png' % output_file_base_name
        l_im_out[0].save(filename)
    
    return filename



# QR CODE IMAGES ###################################################################################



def make_qr_image(token_id, url, version=9, scale_factor=12):
    
    bg_color = d_background_color.get(df_meta.loc[token_id, 'backgroundId'])
    c_distance = color_distance_a(bg_color, c_outline)
    fill_color = c_outline if c_distance>220 else bg_color
    back_color = (255, 255, 255) if c_distance<=220 else bg_color

    # version 5 or 6 for smaller skulls, 9 for larger skulls
    qr = qrcode.QRCode(version=version, box_size=1, border=1, error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(url) # Crypto_Skulls, KobeDLincoln
    qr.make(fit=True)
    img_qr = qr.make_image(fill_color=fill_color, back_color=back_color).convert('RGBA')

    # skull
    img_skull_raw = assemble(token_id, 'cryptoskulls_24', None, False)
    img_skull = Image.new(mode='RGBA', size=(26, 26))
    for i, j in product([-1, 0, 1], [-1, 0, 1]):
        img_skull.paste(bg_color, (i+1, j+1), img_skull_raw)
    img_skull.paste(img_skull_raw, (1,1), img_skull_raw)

    # combining
    img_qr_dim, _ = img_qr.size
    paste_pos = int((img_qr_dim - 26) / 2)
    img_qr.paste(img_skull, (paste_pos, paste_pos), img_skull)

    img_qr = img_qr.resize((img_qr_dim*scale_factor, img_qr_dim*scale_factor), Image.NEAREST).convert('RGB')

    print(token_id, img_qr_dim, img_qr.size, bg_color, c_distance)
    display(img_qr)



# EASTER BUNNY IMAGES ###################################################################################



def make_overlay_image(token_id, image_name, resize=None):
    img = cropped_skulls[token_id]
    img_bunny = Image.open('images/%s.png' % image_name)
    img.paste(img_bunny, (0, 0), img_bunny)
    if resize is not None:
        img = img.resize((24*resize, 24*resize), Image.NEAREST).convert('RGB')
    return(img)



def export_overlay(image_name, filepath_template, resize=None, svg=False):
    for token_id in range(0, 10000):
        if token_id % 1000 == 0:
            print('starting ', token_id)
        if token_id in special_tokens:
            continue
        im = make_overlay_image(token_id, image_name, resize=resize)

        if svg:
            produce_svg(im, token_id, filepath_template, bg_visible=True)
        else:
            im.save(filepath_template % token_id)

        

# BASE CONVERSION (for encoding) ###################################################################



# https://stackoverflow.com/a/50261968
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"

def encode(n):
    try:
        return ALPHABET [n]
    except IndexError:
        raise Exception ("cannot encode: %s" % n)

def dec_to_base(dec=0, base=16):
    if dec < base:
        return encode (dec)
    else:
        return dec_to_base (dec // base, base) + encode (dec % base)



# TWITTER HEADER IMAGES ############################################################################



def export_skull_nation_images(kind, filepath_template, start_id=0):
    print('Exporting %s to template, %s' % (kind, filepath_template))
    for token_id in range(start_id, 10000):
        if token_id % 1000 == 0:
            print('starting ', token_id)
        if token_id in special_tokens:
            continue
        im = make_skull_nation_image(token_id, kind)
        im.save(filepath_template % token_id)



def make_skull_nation_image(token_id, kind):

    assert kind in ['CS_Twitter_Header', 'CS_1x1A', 'CS_1x1B']

    bg_color = d_background_color.get(df_meta.loc[token_id, 'backgroundId'])

    c_distance = color_distance_a(bg_color, c_outline)
    # print(c_distance)
    text_color = (255, 255, 255) if c_distance<=100 else c_outline

    color_bones = d_bones_color.get(df_meta.loc[token_id, 'bonesGene'])
    color_skull = d_skull_color.get(df_meta.loc[token_id, 'skullGene'])
    color_hair = d_hair_color.get(df_meta.loc[token_id, 'hairGene'])
    color_eyes = d_eyes_color.get(df_meta.loc[token_id, 'eyesGene'])
    color_beard = d_beard_color.get(df_meta.loc[token_id, 'beardGene'])

    color_accent_1 = color_bones
    if (color_accent_1 is None) or (color_accent_1 == text_color) or (color_distance_a(bg_color, color_accent_1)<50):
        color_accent_1 = color_hair
    if (color_accent_1 is None) or (color_accent_1 == text_color) or (color_distance_a(bg_color, color_accent_1)<50):
        color_accent_1 = color_eyes
    if (color_accent_1 is None) or (color_accent_1 == text_color) or (color_distance_a(bg_color, color_accent_1)<50):
        color_accent_1 = color_skull

    color_accent_2 = color_hair
    if (color_accent_2 is None) or (color_accent_2 in [c_outline, color_accent_1]):
        color_accent_2 = color_eyes
    if (color_accent_2 is None) or (color_accent_2 in [c_outline, color_accent_1]):
        color_accent_2 = color_skull
    if (color_accent_2 is None) or (color_accent_2 in [c_outline, color_accent_1]):
        color_accent_2 = color_beard
    if (color_accent_2 is None) or (color_accent_2 in [c_outline, color_accent_1]):
        color_accent_2 = bg_color

    color_accent_3 = color_eyes
    if (color_accent_3 is None) or (color_accent_3 in [c_outline, color_accent_1, color_accent_2]):
        color_accent_3 = color_beard
    if (color_accent_3 is None) or (color_accent_3 in [c_outline, color_accent_1, color_accent_2]):
        color_accent_3 = bg_color
    if (color_accent_3 is None) or (color_accent_3 in [c_outline, color_accent_1, color_accent_2]):
        color_accent_3 = color_skull

    if kind == 'CS_Twitter_Header':

        img = Image.new(mode='RGBA', size=(1500, 500), color=bg_color)
        d1 = ImageDraw.Draw(img)
        font = ImageFont.truetype('font/cryptoskulls.otf', 78)
        d1.text((1342, 247), 'CS', fill=text_color, font=font)
        d1.text((1342, 285), str(token_id).rjust(4, '0'), fill=text_color, font=font)

        d1.rectangle(((1166, 270), (1199, 303)), fill=color_accent_2)
        d1.rectangle(((1166, 304), (1199, 337)), fill=color_accent_1)
        d1.rectangle(((1200, 270), (1233, 303)), fill=c_outline)
        d1.rectangle(((1200, 304), (1233, 337)), fill=color_skull)

        # polygons (three positions for each x)
        str_x_enc = dec_to_base(token_id, 3).rjust(9, '0')
        xs = [225, 330, 426, 698, 793, 1046, 735, 834, 1083]
        for i, str_x in enumerate(str_x_enc):
            xs[i] += 25 * (int(str_x) - 1)
        d1.polygon(((xs[0],0), (xs[0],25), (xs[1],25), (xs[1],50), (xs[2],50), (xs[2],75),
                    (530,75), (530,25), (xs[3],25), (xs[3],50), (xs[4],50), (xs[4],75),
                    (922,75), (922,25), (xs[5],25), (xs[5],0)), fill=color_accent_1)
        d1.polygon(((xs[6],499), (xs[6],474), (xs[7],474), (xs[7],424),
                    (988,424), (988,449), (xs[8],449), (xs[8],474),
                    (1138,474), (1138,499)), fill=color_accent_1)

        # overlay image
        img_text_file = 'images/header_light.png' if c_distance<=100 else 'images/header_dark.png'
        img_text = Image.open(img_text_file)
        img.paste(img_text, (0, 0), img_text)

    elif kind == 'CS_1x1A':

        img = Image.new(mode='RGB', size=(1000, 1000), color=bg_color)

        img_skull = cropped_skulls[token_id]
        img_skull = img_skull.resize((34*24, 34*24), Image.NEAREST)
        xy = int((img.size[0] - img_skull.size[0]) / 2)
        img.paste(img_skull, (xy, xy))

        d1 = ImageDraw.Draw(img)
        font = ImageFont.truetype('font/cryptoskulls.otf', 70)
        d1.text((858, 400), 'CS', fill=text_color, font=font)
        d1.text((858, 438), str(token_id).rjust(4, '0'), fill=text_color, font=font)
        font = ImageFont.truetype('font/apercu_mono.otf', 16)
        d1.text((698, 925), str(token_id).rjust(4, '0'), fill=text_color, font=font)

        x=892
        y=194
        d1.rectangle(((x, y), (x+33, y+33)), fill=color_accent_2)
        d1.rectangle(((x, y+34), (x+33, y+34+33)), fill=color_accent_1)
        d1.rectangle(((x+34, y), (x+34+33, y+33)), fill=c_outline)
        d1.rectangle(((x+34, y+34), (x+34+33, y+34+33)), fill=color_skull)
        y=605
        d1.rectangle(((x+34, y), (x+34+33, y+33)), fill=c_outline)

        # polygons (three positions for each x)
        d1.polygon(((0, 195), (32,195), (32,362), (100,362), (100,536), (66,536),
                    (66,663), (32,663), (32,738), (0,738)), fill=color_accent_1)

        # overlay image
        img_text_file = 'images/square_light.png' if c_distance<=100 else 'images/square_dark.png'
        img_text = Image.open(img_text_file)
        img.paste(img_text, (0, 0), img_text)

    elif kind == 'CS_1x1B':

        # c_distance_1 = color_distance_a(color_accent_1, c_outline)
        # c_distance_2 = color_distance_a(color_accent_2, c_outline)

        # text_color_1 = (255, 255, 255) if c_distance_1<=100 else c_outline
        # text_color_2 = (255, 255, 255) if c_distance_2<=100 else c_outline

        img = Image.new(mode='RGB', size=(1000, 1000), color=color_accent_2)
        d1 = ImageDraw.Draw(img)
        d1.rectangle(((0, 501), (1000, 1000)), fill=color_accent_1)

        img_skull = cropped_skulls[token_id]
        img_skull = img_skull.resize((21*24, 21*24), Image.NEAREST)
        y = int((img.size[0] - img_skull.size[0]) / 2)
        img.paste(img_skull, (0, y))

        font = ImageFont.truetype('font/cryptoskulls.otf', 190)
        d1.text((652, 201), 'CS', fill=text_color, font=font)
        d1.text((652, 296), str(token_id).rjust(4, '0'), fill=text_color, font=font)
        font = ImageFont.truetype('font/apercu_mono.otf', 16)
        d1.text((698, 925), str(token_id).rjust(4, '0'), fill=text_color, font=font)

        x=900
        y=178
        d1.rectangle(((x, y), (x+33, y+33)), fill=color_accent_3)
        d1.rectangle(((x, y+34), (x+33, y+34+33)), fill=color_accent_1)
        d1.rectangle(((x+34, y), (x+34+33, y+33)), fill=c_outline)
        d1.rectangle(((x+34, y+34), (x+34+33, y+34+33)), fill=color_skull)
        x=892
        y=605
        d1.rectangle(((x+34, y), (x+34+33, y+33)), fill=c_outline)

        # overlay image
        # img_text_file_1 = 'images/square_light.png' if c_distance_1<=100 else 'images/square_dark.png'
        # img_text_1 = Image.open(img_text_file_1).crop((0, 501, 1000, 1000))
        # img.paste(img_text_1, (0, 501), img_text_1)

        # img_text_file_2 = 'images/square_light.png' if c_distance_2<=100 else 'images/square_dark.png'
        # img_text_2 = Image.open(img_text_file_2).crop((0, 0, 1000, 500))
        # img.paste(img_text_2, (0, 0), img_text_2)

        img_text_file = 'images/square_light.png' if c_distance<=100 else 'images/square_dark.png'
        img_text = Image.open(img_text_file)
        img.paste(img_text, (0, 0), img_text)


        # img = Image.blend(img, Image.open('images/10.jpg'), 0.25)

    return img


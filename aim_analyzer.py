import lzma
import struct
from datetime import datetime, timezone, timedelta
from ossapi import *
import requests
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats
import io
import base64

api = OssapiV2(19003, 'TVMi13ZOzWLvCo1CUMTg4Bs49r0ytI1PZ0INDr0d')

PLAYING_AREA_CENTER = (256, 192)


def get_string_length(binarystream, offset):
    result = 0
    shift = 0
    while True:
        byte = binarystream[offset]
        offset += 1
        result = result | ((byte & 0b01111111) << shift)
        if (byte & 0b10000000) == 0x00:
            break
        shift += 7
    return result, offset


def unpack_once(specifier, replay_data, offset):
    specifier = f"<{specifier}"
    unpacked = struct.unpack_from(specifier, replay_data, offset)
    offset += struct.calcsize(specifier)
    return unpacked[0], offset


def unpack_string(replay_data, offset):
    if replay_data[offset] == 0x00:
        offset += 1
    else:
        offset += 1
        string_length, offset = get_string_length(replay_data, offset)
        offset_end = offset + string_length
        string = replay_data[offset:offset_end].decode("utf-8")
        offset = offset_end
        return string, offset


def unpack_timestamp(replay_data, offset):
    ticks, offset = unpack_once("q", replay_data, offset)
    timestamp = datetime.min + timedelta(microseconds=ticks/10)
    timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp, offset


def unpack_play_data(replay_data, offset):
    replay_length, offset = unpack_once("i", replay_data, offset)
    offset_end = offset + replay_length
    data = replay_data[offset:offset_end]
    data = lzma.decompress(data, format=lzma.FORMAT_AUTO)
    data = data.decode("ascii")
    (replay_data, rng_seed) = parse_replay_data(data)
    offset = offset_end
    return (replay_data, rng_seed), offset


def parse_replay_data(replay_data_str):
    replay_data_str = replay_data_str.rstrip(",")
    events = [event.split('|') for event in replay_data_str.split(',')]

    rng_seed = None
    play_data = []
    for event in events:
        time_delta = int(event[0])
        x = event[1]
        y = event[2]
        keys = int(event[3])

        if time_delta == -12345 and event == events[-1]:
            rng_seed = keys
            continue

        event = [time_delta, float(x), float(y), keys]

        play_data.append(event)

    return (play_data, rng_seed)


def unpack_life_bar(replay_data, offset):
    life_bar, offset = unpack_string(replay_data, offset)
    if not life_bar:
        return 0, offset

    life_bar = life_bar[:-1]
    states = [state.split("|") for state in life_bar.split(",")]

    return [(int(s[0]), float(s[1])) for s in states], offset


def unpack_replay_id(data, offset):
    try:
        replay_id = unpack_once("q", data, offset)
    except struct.error:
        replay_id = unpack_once("l", data, offset)
    return replay_id


FUNCTION_FACTORY = {
    "string": unpack_string,
    "life_bar": unpack_life_bar,
    "timestamp": unpack_timestamp,
    "replay_data": unpack_play_data,
    "replay_id": unpack_replay_id
}


def replay_file_to(file):
    with open(file, "rb") as osr_file:
        data = osr_file.read()
    offset = 0

    unpack_formats = {
        "mode": "b",
        "version": "i",
        "beatmap": "string",
        "username": "string",
        "replay_hash": "string",
        "count_300": "h",
        "count_100": "h",
        "count_50": "h",
        "count_geki": "h",
        "count_katu": "h",
        "count_miss": "h",
        "total_score": "i",
        "max_combo": "h",
        "is_perfect": "?",
        "mods": "i",
        "life_bar_graph": "life_bar",
        "timestamp": "timestamp",
        "replay_data": "replay_data",
        "replay_id": "replay_id",
    }

    results_dict = dict()

    for info in unpack_formats:
        if unpack_formats[info] in FUNCTION_FACTORY:
            value, offset = FUNCTION_FACTORY[unpack_formats[info]](
                data, offset)
        else:
            value, offset = unpack_once(unpack_formats[info], data, offset)
        results_dict[info] = value

    return results_dict


def is_in_circle(hit_object, cursor, cs):
    return ((hit_object[0] - cursor[0]) ** 2 + (hit_object[1] - cursor[1]) ** 2) ** (1 / 2) <= (54.4 - 4.48 * cs)


def distance(hit_object, cursor):
    return ((hit_object[0] - cursor[0]) ** 2 + (hit_object[1] - cursor[1]) ** 2) ** (1 / 2)


def is_in_hit_window(od, hit_object, cursor, speed=1):
    return abs(10 * (od - 19.95)) >= abs(hit_object[2] - cursor[2])


def before_hit_window(od, hit_object, cursor, speed=1):
    return not is_in_hit_window(od, hit_object, cursor, speed=1) and hit_object[2] > cursor[2]


def after_hit_window(od, hit_object, cursor, speed=1):
    return not is_in_hit_window(od, hit_object, cursor, speed=1) and hit_object[2] < cursor[2]


def matches(hit_objects, cursors, od, cs):
    matched_pairs = []
    missed_pairs = []
    potential_miss_cursor = None
    object_idx, cursor_idx = 0, 0
    while object_idx < len(hit_objects) and cursor_idx < len(cursors):
        if is_in_hit_window(od, hit_objects[object_idx], cursors[cursor_idx]):
            if is_in_circle(hit_objects[object_idx], cursors[cursor_idx], cs):
                matched_pairs.append([hit_objects[object_idx][0], hit_objects[object_idx]
                                     [1], cursors[cursor_idx][0], cursors[cursor_idx][1]])
                object_idx += 1
                cursor_idx += 1
            else:
                potential_miss_cursor = cursors[cursor_idx]
                cursor_idx += 1
        elif after_hit_window(od, hit_objects[object_idx], cursors[cursor_idx]):
            if cursor_idx == len(cursors) - 1:
                cursor_idx += 1
            elif potential_miss_cursor is not None:
                missed_pairs.append([hit_objects[object_idx][0], hit_objects[object_idx]
                                    [1], potential_miss_cursor[0], potential_miss_cursor[1]])
                object_idx += 1
                potential_miss_cursor = None
            else:
                missed_pairs.append([hit_objects[object_idx][0], hit_objects[object_idx]
                                    [1], cursors[cursor_idx][0], cursors[cursor_idx][1]])
                object_idx += 1
        elif before_hit_window(od, hit_objects[object_idx], cursors[cursor_idx]):
            cursor_idx += 1

    return matched_pairs, missed_pairs


def angle_finder(x, y):
    x_center = x - PLAYING_AREA_CENTER[0]
    y_center = PLAYING_AREA_CENTER[1] - y

    return math.degrees(math.atan2(y_center, x_center))


def angle_diff(x_circle, y_circle, x_aim, y_aim):
    unit_vector_1 = (x_circle, y_circle) / np.linalg.norm((x_circle, y_circle))
    unit_vector_2 = (x_aim, y_aim) / np.linalg.norm((x_aim, y_aim))
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)

    return angle


def aim_diff(x_circle, y_circle, x_aim, y_aim):
    circle_to_center = distance((x_circle, y_circle), PLAYING_AREA_CENTER)
    cursor_to_center = distance((x_aim, y_aim), PLAYING_AREA_CENTER)

    return circle_to_center - cursor_to_center


def analyze_aim(file):
    replay = replay_file_to(file)
    username = replay['username']
    beatmap = api.beatmap(checksum=replay['beatmap'])
    beatmap_name = beatmap._beatmapset.title
    beatmap_diff = beatmap.version
    beatmap_id = beatmap.id

    raw_beatmap = requests.session().get(
        f"https://kitsu.moe/api/osu/{beatmap_id}").content

    hit_objects = raw_beatmap.decode("utf-8") .split('[HitObjects]')[1]

    hitobject_info = np.array([hitobject.split(
        ',')[0:3] for hitobject in hit_objects.split("\r\n")[1:-1]]).astype(int)

    replay_info = replay['replay_data'][0]

    replay_array = np.array(replay_info)

    replay_array = np.hstack(
        (replay_array, np.cumsum(replay_array[:, 0]).reshape(-1, 1)))

    current = 0
    unique_replay_array = np.empty((0, 5))
    for replay_index in replay_array:
        current_press = replay_index[3]
        if current_press != current:
            current = current_press
            if current_press != 0:
                unique_replay_array = np.vstack(
                    (unique_replay_array, replay_index))

    unique_replay_array = unique_replay_array[:, [1, 2, 4]]
    cs = beatmap.cs
    od = beatmap.accuracy

    matched_pairs, missed_pairs = matches(
        hitobject_info, unique_replay_array, od, cs)

    pair_coords = [(pair[2] - pair[0], pair[3] - pair[1])
                   for pair in matched_pairs]
    missed_coords = [(pair[2] - pair[0], pair[3] - pair[1])
                     for pair in missed_pairs]

    centroid = [sum(x)/len(x) for x in zip(*pair_coords+missed_coords)]

    img1 = io.BytesIO()

    plt.clf()
    plt.scatter(*zip(*pair_coords), c='lightgreen', s=5)
    if len(missed_coords) > 0:
        plt.scatter(*zip(*missed_coords), c='red', s=5)
    plt.title(f'Aim Analysis for {username} on {beatmap_diff} difficulty of {beatmap_name}')
    plt.scatter(centroid[0], centroid[1], c='black')
    plt.axis('equal')
    plt.axis('off')
    theta = np.linspace(0, 2*np.pi, 100)
    r = (54.4 - 4.48 * cs)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    plt.plot(x, y)
    plt.savefig(img1, format='png')
    img1.seek(0)

    img1 = base64.b64encode(img1.getvalue()).decode()

    matched_pair_angles = [angle_finder(*pair[0:2]) for pair in matched_pairs]

    aim_difference = [aim_diff(*pair) for pair in matched_pairs]
    angle_difference = [angle_diff(*pair) for pair in matched_pairs]

    eighths_aim = [[] for _ in range(8)]
    eighths_angle = [[] for _ in range(8)]

    non_negative_angles = [i + 360 if i <
                           0 else i for i in matched_pair_angles]

    for i in range(len(non_negative_angles)):
        eighth_location = round(non_negative_angles[i]/(360/8)) % 8
        eighths_aim[eighth_location].append(aim_difference[i])
        eighths_angle[eighth_location].append(angle_difference[i])

    eighths_aim_mean = [np.mean(eighth) for eighth in eighths_aim]
    eighths_angle_skew = [scipy.stats.skew(eighth) for eighth in eighths_angle]

    img2 = io.BytesIO()
    plt.clf()
    fig, ax = plt.subplots(1)
    ax.set_title('Under/Over Aiming')

    ax.set_aspect(384/512, adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(.5, c='black')
    ax.axvline(.5, c='black')
    ax.plot([0, 1], [0, 1], 'black')
    ax.plot([1, 0], [0, 1], 'black')
    x = .1*np.cos(theta) + .5
    y = .1*np.sin(theta) + .5
    ax.plot(x, y)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    for j in range(8):
        eighths = eighths_aim
        ins = ax.inset_axes([np.cos((j + 0.5) * (2 * np.pi / 8)) / 2.5 +
                             0.4, np.sin((j + 0.5) * (2 * np.pi / 8)) / 2.5 + 0.4, 0.2, 0.2])
        ins.set_xticks([])
        ins.set_yticks([])
        ins.hist(eighths[j])

        description = round(
            eighths_aim_mean[j], 2)
        x_arrow = .1*np.cos(2 * np.pi / 8 * (j + 0.5)) + 0.5
        y_arrow = .1*np.sin(2 * np.pi / 8 * (j + 0.5)) + 0.5
        dx = max(min(description * np.cos(2 * np.pi / 8 * (j + 0.5)) / 50, 5), -5)
        dy = max(min(description * np.sin(2 * np.pi / 8 * (j + 0.5)) / 50, 5), -5)

        ax.arrow(x_arrow, y_arrow, dx, dy, width=.003)

        ax.text(np.cos((j + 0.5) * (2 * np.pi / 8)) / 4.5 + 0.5, np.sin(
            (j + 0.5) * (2 * np.pi / 8)) / 4.5 + 0.48, description, ha='center')

        ins.axvline(0, c='black')

    plt.savefig(img2, format='png', dpi=200)
    img2.seek(0)

    img2 = base64.b64encode(img2.getvalue()).decode()

    img3 = io.BytesIO()
    plt.clf()
    fig, ax = plt.subplots(1)
    ax.set_title('Angular Misaiming')

    ax.set_aspect(384/512, adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(.5, c='black')
    ax.axvline(.5, c='black')
    ax.plot([0, 1], [0, 1], 'black')
    ax.plot([1, 0], [0, 1], 'black')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    for j in range(8):
        eighths = eighths_angle
        ins = ax.inset_axes([np.cos((j + 0.5) * (2 * np.pi / 8)) / 2.5 +
                             0.4, np.sin((j + 0.5) * (2 * np.pi / 8)) / 2.5 + 0.4, 0.2, 0.2])
        ins.set_xticks([])
        ins.set_yticks([])
        ins.hist(eighths[j])

        description = round(
            eighths_angle_skew[j], 2)
        ax.text(np.cos((j + 0.5) * (2 * np.pi / 8)) / 4.5 + 0.5, np.sin(
            (j + 0.5) * (2 * np.pi / 8)) / 4.5 + 0.48, description, ha='center')
    plt.savefig(img3, format='png', dpi=200)
    img3.seek(0)

    img3 = base64.b64encode(img3.getvalue()).decode()

    return "data:image/png;base64,{}".format(img1), "data:image/png;base64,{}".format(img2), "data:image/png;base64,{}".format(img3)

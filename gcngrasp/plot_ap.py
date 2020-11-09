import argparse
import pickle
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import operator
from sklearn.metrics import average_precision_score
from collections import defaultdict

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '../'))
from visualize import mkdir

def get_results_dir(log_dir, results_dir):
    dirs = os.listdir(log_dir)
    for dir in dirs:
        if dir.find(results_dir) >= 0:
            return dir
    raise FileNotFoundError('Unable to find appropriate folder for experiment {}'.format(results_dir))

def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])

def get_ap_random(n_seeds, labels):
    aps = []
    N = len(labels)
    for i in range(n_seeds):
        np.random.seed(i)
        random_probs = np.random.uniform(low=0, high=1, size=(N)).tolist()
        ap = average_precision_score(labels, random_probs)
        aps.append(ap)
    return np.mean(aps)


def draw_plot(dictionary, dictionary_random, n_elems, window_title, plot_title, x_label, output_path, to_show, plot_color, mAP_random=None):
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    # 

    sorted_values_max = []
    sorted_values_random = []
    sorted_values_starts = []
    sorted_values_widths = []
    for key in sorted_keys:
        val_rand = dictionary_random[key]
        val_prob = dictionary[key]
        width = val_prob - val_rand
        # assert width >= 0.0
        sorted_values_random.append(val_rand)
        sorted_values_starts.append(val_rand)
        sorted_values_widths.append(width)
        sorted_values_max.append(max(val_rand, val_prob))

    # if output_path.find('all_task') >= 0:
    #     embed()
    sorted_values_random = tuple(sorted_values_random)
    sorted_values_starts = tuple(sorted_values_starts)
    sorted_values_widths = tuple(sorted_values_widths)
    sorted_values_max = tuple(sorted_values_max)

    plt.barh(range(n_elems), sorted_values_random, color='red')
    plt.barh(range(n_elems), sorted_values_widths, left=sorted_values_starts, color=plot_color)
    """
        Write number on side of bar
    """
    fig = plt.gcf() # gcf - get current figure
    axes = plt.gca()
    r = fig.canvas.get_renderer()
    for i, val in enumerate(sorted_values_max):
        str_val = " " + str(val) # add a space before
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val+0.02, i, str_val, color=plot_color, va='center', fontweight='bold')
        # re-set axes to show number inside the figure
        if i == (len(sorted_values_max)-1): # largest bar
            adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_elems), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_elems * (tick_font_size * 1.4) # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height 
    top_margin = 0.15 # in percentage of the figure height
    bottom_margin = 0.05 # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)


    if mAP_random:
        assert mAP_random >= 0 and mAP_random <= 1.0
        plt.axvline(x=mAP_random, color='k', linestyle='--')
    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GCN training")
    parser.add_argument('--base_dir', default='', help='Location of dataset', type=str)
    parser.add_argument('--log_dir', default='', help='Location of pretrained checkpoint models', type=str)
    parser.add_argument('--save_dir', default='', type=str)
    args = parser.parse_args()

    if args.base_dir != '':
        if not os.path.exists(args.base_dir):
            raise FileNotFoundError(
                'Provided base dir {} not found'.format(
                    args.base_dir))
    else:
        args.base_dir = os.path.join(os.path.dirname(__file__), '../data')

    if args.log_dir != '':
        if not os.path.exists(args.log_dir):
            raise FileNotFoundError(
                'Provided checkpoint dir {} not found'.format(
                    args.log_dir))
    else:
        args.log_dir = os.path.join(os.path.dirname(__file__), '../checkpoints')

    folder_dir = 'taskgrasp'
    _, _, _, map_obj2class = pickle.load(
        open(os.path.join(args.base_dir, folder_dir, 'misc.pkl'), 'rb'))

    if args.save_dir == '':
        args.save_dir = os.path.join(os.path.dirname(__file__), '../results')

    # exp_title = 'Held-out Tasks GCNGrasp'
    # data = {
    #     "0":"gcngrasp_split_mode_t_split_idx_0",
    #     "1":"gcngrasp_split_mode_t_split_idx_1",
    #     "2":"gcngrasp_split_mode_t_split_idx_2",
    #     "3":"gcngrasp_split_mode_t_split_idx_3",
    # }
    # exp_name = 'gcngrasp_t'

    exp_title = 'Held-out Objects GCNGrasp'
    data = {
        "0":"gcngrasp_split_mode_o_split_idx_0",
        "1":"gcngrasp_split_mode_o_split_idx_1",
        "2":"gcngrasp_split_mode_o_split_idx_2",
        "3":"gcngrasp_split_mode_o_split_idx_3",
    }
    exp_name = 'gcngrasp_o'
    
    args.save_dir = os.path.join(args.save_dir, exp_name)
    mkdir(args.save_dir)

    merged_task_ap = {}
    merged_class_ap = {}
    merged_obj_ap = {}
    merged_task_ap_random = {}
    merged_class_ap_random = {}
    merged_obj_ap_random = {}

    merged_task_labels = []
    merged_class_labels = []
    merged_obj_labels = []

    # Plotting information
    window_title = "mAP"
    x_label = "Average Precision"
    to_show = False
    plot_color = 'royalblue'

    for split_idx, split_dir in data.items():
        pkl_file = get_results_dir(args.log_dir, split_dir)
        print('Loading {} results from {}'.format(split_idx, pkl_file))
        pkl_file = os.path.join(args.log_dir, pkl_file, 'results2_ap', 'results.pkl')
        results = pickle.load(open(pkl_file, 'rb'))

        obj_ap = defaultdict(list)
        obj_ap_random = defaultdict(list) 
        obj_probs = defaultdict(list)
        obj_labels = defaultdict(list)

        class_ap = defaultdict(list)
        class_ap_random = defaultdict(list)
        class_probs = defaultdict(list)
        class_labels = defaultdict(list)

        task_ap = defaultdict(list)
        task_ap_random = defaultdict(list)
        task_probs = defaultdict(list)
        task_labels = defaultdict(list)

        preds = results['preds']
        probs = results['probs']
        labels = results['labels']

        for obj in probs.keys():
            if type(obj) != tuple:
                for task in probs[obj].keys():
                    if type(task) == str:
                        assert len(probs[obj][task]) == len(labels[obj][task])
                        obj_probs[obj] += probs[obj][task]
                        obj_labels[obj] += labels[obj][task]

                        task_probs[task] += probs[obj][task]
                        task_labels[task] += labels[obj][task]

                        obj_class = map_obj2class[obj]
                        class_probs[obj_class] += probs[obj][task]
                        class_labels[obj_class] += labels[obj][task]

        for obj in obj_probs.keys():
            obj_prob = obj_probs[obj]
            obj_label = obj_labels[obj]
            merged_obj_labels += obj_labels[obj]
            assert len(obj_prob) == len(obj_label)
            ap_random = get_ap_random(5, obj_label)
            ap = average_precision_score(obj_label, obj_prob)
            if not np.isnan(ap):
                obj_ap[obj] = ap
                obj_ap_random[obj] = ap_random

        for task in task_probs.keys():
            task_prob = task_probs[task]
            task_label = task_labels[task]
            merged_task_labels += task_labels[task]
            assert len(task_prob) == len(task_label)
            ap_random = get_ap_random(5, task_label)
            ap = average_precision_score(task_label, task_prob)
            if not np.isnan(ap):
                task_ap[task] = ap
                task_ap_random[task] = ap_random

        for obj_class in class_probs.keys():
            class_prob = class_probs[obj_class]
            class_label = class_labels[obj_class]
            merged_class_labels += class_labels[obj_class]
            assert len(class_prob) == len(class_label)
            ap_random = get_ap_random(5, class_label)
            ap = average_precision_score(class_label, class_prob)
            if not np.isnan(ap):
                class_ap[obj_class] = ap
                class_ap_random[obj_class] = ap_random

        # held out instances
        obj_ap = dict(obj_ap)
        obj_ap_random = dict(obj_ap_random)
        mAP = np.mean(list(obj_ap.values()))
        n_elems = len(list(obj_ap.keys()))
        plot_title = "mAP = {0:.2f}%".format(mAP*100)
        output_path = os.path.join(args.save_dir, "{}_instance_mAP.png".format(split_idx))
        draw_plot(
            obj_ap,
            obj_ap_random,
            n_elems,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color
            )
        plt.clf()

        # held out task
        task_ap = dict(task_ap)
        task_ap_random = dict(task_ap_random)
        mAP = np.mean(list(task_ap.values()))
        n_elems = len(list(task_ap.keys()))
        plot_title = "mAP = {0:.2f}%".format(mAP*100)
        output_path = os.path.join(args.save_dir, "{}_task_mAP.png".format(split_idx))
        draw_plot(
            task_ap,
            task_ap_random,
            n_elems,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color
            )

        # held out class
        class_ap = dict(class_ap)
        class_ap_random = dict(class_ap_random)
        mAP = np.mean(list(class_ap.values()))
        n_elems = len(list(class_ap.keys()))
        plot_title = "mAP = {0:.2f}%".format(mAP*100)
        output_path = os.path.join(args.save_dir, "{}_class_mAP.png".format(split_idx))
        draw_plot(
            class_ap,
            class_ap_random,
            n_elems,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color
            )

        merged_task_ap = {**merged_task_ap, **task_ap}
        merged_class_ap = {**merged_class_ap, **class_ap}
        merged_obj_ap = {**merged_obj_ap, **obj_ap}

        merged_task_ap_random = {**merged_task_ap_random, **task_ap_random}
        merged_class_ap_random = {**merged_class_ap_random, **class_ap_random}
        merged_obj_ap_random = {**merged_obj_ap_random, **obj_ap_random}

    # held out instance
    mAP = np.mean(list(merged_obj_ap.values()))
    n_elems = len(list(merged_obj_ap.keys()))
    plot_title = "mAP = {0:.2f}%".format(mAP*100)
    output_path = os.path.join(args.save_dir, "{}_instance_mAP.png".format('all'))
    draw_plot(
        merged_obj_ap,
        merged_obj_ap_random,
        n_elems,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color
        )
    plt.clf()


    # held out task
    mAP = np.mean(list(merged_task_ap.values()))
    n_elems = len(list(merged_task_ap.keys()))
    plot_title = "mAP = {0:.2f}%".format(mAP*100)
    output_path = os.path.join(args.save_dir, "{}_task_mAP.png".format('all'))
    draw_plot(
        merged_task_ap,
        merged_task_ap_random,
        n_elems,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        )

    # held out class
    mAP = np.mean(list(merged_class_ap.values()))
    n_elems = len(list(merged_class_ap.keys()))
    plot_title = "mAP = {0:.2f}%".format(mAP*100)
    output_path = os.path.join(args.save_dir, "{}_class_mAP.png".format('all'))
    draw_plot(
        merged_class_ap,
        merged_class_ap_random,
        n_elems,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color
        )
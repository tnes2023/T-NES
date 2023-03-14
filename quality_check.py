import soundfile as sf
from pypesq import pesq
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as color


def get_mean_score(folder):
    # folder = "nes_200_0"
    audios = os.listdir(os.path.join('generated_audios', folder))
    audios.sort()
    audio_groups = {}

    for audio in audios:
        idx = audio.split('-')[0]
        if idx not in audio_groups.keys():
            audio_groups[idx] = [audio]
        else:
            audio_groups[idx].append(audio)

    avg_score = {6:[]}
    for id, audios in audio_groups.items():
        # print(id)
        ref, sr_ref = sf.read(os.path.join('generated_audios', folder, audios[-1]))
        avg_score[6].append(pesq(ref, ref, sr_ref))
        for i in range(0, len(audios)-1):
            deg, sr_deg = sf.read(os.path.join('generated_audios', folder, audios[i]))
            score = pesq(ref, deg, sr_deg)
            if i not in avg_score.keys():
                avg_score[i] = [score]
            else:
                avg_score[i].append(score)

    mean_score = []
    mean_score.append(np.mean(avg_score[6]))
    for i in range(6):
        scores = avg_score[i]

        # print(i*0.1, np.mean(scores))
        mean_score.append(np.mean(scores))

    return mean_score

COLORS = [color.TABLEAU_COLORS['tab:blue'], color.TABLEAU_COLORS['tab:orange'],
          color.TABLEAU_COLORS['tab:green'], color.TABLEAU_COLORS['tab:red'],
          color.TABLEAU_COLORS['tab:purple'], color.TABLEAU_COLORS['tab:brown'],
          color.TABLEAU_COLORS['tab:pink'], color.TABLEAU_COLORS['tab:gray'],
          color.TABLEAU_COLORS['tab:olive'], color.TABLEAU_COLORS['tab:cyan']]


def human_study_plot(axes):
    plt.figure(figsize=(8,6))
    data = {'Increased Character Error Rates (ICER)':['original', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'T-NES': [5, 3.1, 3.2, 3, 3.05, 3.05, 3.05],
            'ZOO-Adam': [5, 3.05, 3.1, 3.26, 3.00,3.00,3.05 ]}
    df = pd.DataFrame(data)
    df = pd.melt(df,  id_vars=['Increased Character Error Rates (ICER)'], value_vars=['T-NES','ZOO-Adam'],var_name='Method', value_name='Score')
    sns.barplot(x='Increased Character Error Rates (ICER)', y='Score', hue="Method", data=df, ax=axes[1], ylim=(1,5))
    axes[1].set_title("User Study")

    for item in ([axes[1].title, axes[1].xaxis.label, axes[1].yaxis.label] +
                 axes[1].get_xticklabels() + axes[1].get_yticklabels()):
        item.set_fontsize(20)  # fontsize of the axes title
    plt.setp(axes[1].get_legend().get_texts(), fontsize='20')  # for legend text
    plt.setp(axes[1].get_legend().get_title(), fontsize='20')  # for legend title
    # plt.tight_layout()
    # plt.show()
    # axes[1].set(xlabel=None)
    axes[1].set()

def pesq_plot():
    # nes_0 = get_mean_score("nes_200_0")
    sns.set_theme()

    plt.figure(figsize=(8,6))

    nes_1 = get_mean_score("nes_200_1")
    adam_0 = get_mean_score("zoo_adam_200_0")

    df2 = pd.DataFrame()
    df2['Increased Character Error Rates (ICER)'] = ['original', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    df2['PESQ'] = nes_1
    df2['Method'] = 'T-NES'

    df3 = pd.DataFrame()
    df3['Increased Character Error Rates (ICER)'] = ['original', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    df3['PESQ'] = adam_0
    df3['Method'] = 'ZOO-Adam'

    # df4 = pd.DataFrame()
    # df4['increased_cer'] = ['original', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # df4['quality'] = newton_1
    # df4['method'] = 'Zoo-newton without TD'
    def change_width(ax, new_value) :
        for patch in ax.patches :
            current_width = patch.get_width()
            diff = current_width - new_value

            # we change the bar width
            patch.set_width(new_value)

            # we recenter the bar
            patch.set_x(patch.get_x() + diff * .5)

    df = pd.concat([df2, df3])
    sns.barplot(x="Increased Character Error Rates (ICER)", y="PESQ", hue="Method", data=df, ylim=(1,5))

    # for i, p in enumerate(ax.patches):
    #         ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
    #                     ha='center', va='bottom',
    #                     color= 'black')

    # change_width(ax, .6)
    plt.title("Perceptual Evaluation of Speech Quality (PESQ)")
    # for item in ([plt.title, plt.xaxis.label, plt.yaxis.label] +
    #              plt.get_xticklabels() + plt.get_yticklabels()):
    #     item.set_fontsize(20)  # fontsize of the axes title
    # plt.setp(plt.get_legend().get_texts(), fontsize='20')  # for legend text
    # plt.setp(plt.get_legend().get_title(), fontsize='20')  # for legend title
    # axes[0].set(xlabel=None)
    plt.tight_layout()
    plt.show()


def subplots():
    # sns.set_theme()
    sns.set()
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True)
    # nes_1 = get_mean_score("nes_200_1")
    # adam_0 = get_mean_score("zoo_adam_200_0")

    nes_1 = [4.5, 2.278437300732261, 2.275510787963867, 2.2452136909260467, 2.236982300877571, 2.241751636777605, 2.3452828923861184]
    adam_0 = [4.5, 2.282963652359812, 2.267765760421753, 2.2783581097920735, 2.2578710118929544, 2.226555848121643, 2.3438676834106444]
    df2 = pd.DataFrame()
    df2['Increased Character Error Rates (ICERs)'] = ['original', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    df2['PESQ'] = nes_1
    df2['Method'] = 'T-NES'

    df3 = pd.DataFrame()
    df3['Increased Character Error Rates (ICERs)'] = ['original', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    df3['PESQ'] = adam_0
    df3['Method'] = 'ZOO-Adam'

    df = pd.concat([df2, df3])
    sns.barplot(x="Increased Character Error Rates (ICERs)", y="PESQ", hue="Method", data=df, ax=axes[0])
    axes[0].set_title('PESQ')
    # plt.figure(figsize=(8,6))
    data = {'Increased Character Error Rates (ICERs)':['original', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'T-NES': [4.7, 3.1, 3.2, 3, 3.05, 3.05, 3.05],
            'ZOO-Adam': [4.8, 3.05, 3.1, 3.26, 3.00,3.00,3.05]}
    df = pd.DataFrame(data)
    df = pd.melt(df,  id_vars=['Increased Character Error Rates (ICERs)'], value_vars=['T-NES','ZOO-Adam'],var_name='Method', value_name='Score')
    sns.barplot(x='Increased Character Error Rates (ICERs)', y='Score', hue="Method", data=df, ax=axes[1])

    axes[1].set_title("User Study")
    axes[0].set_title("Perceptual Evaluation of Speech Quality (PESQ)")
    # axes[1].set_ylim(1,5)
    axes[1].set_yticklabels([" ",1,2,3,4, 5])
    for item in ([axes[0].title, axes[0].xaxis.label, axes[0].yaxis.label] +
                 axes[0].get_xticklabels() + axes[0].get_yticklabels()):
        item.set_fontsize(24)  # fontsize of the axes title
    plt.setp(axes[0].get_legend().get_texts(), fontsize='20')  # for legend text
    plt.setp(axes[0].get_legend().get_title(), fontsize='20')  # for legend title

    for item in ([axes[1].title, axes[1].xaxis.label, axes[1].yaxis.label] +
                 axes[1].get_xticklabels() + axes[1].get_yticklabels()):
        item.set_fontsize(24)  # fontsize of the axes title
    plt.setp(axes[1].get_legend().get_texts(), fontsize='20')  # for legend text
    plt.setp(axes[1].get_legend().get_title(), fontsize='20')  # for legend title

    plt.tight_layout()
    # plt.savefig('plot/quality2.pdf')
    plt.show()
    # sns.set_theme()
    # human_study_plot(axes)
    # pesq_plot(axes)
    # plt.xlabel("Increased Character Error Rates (ICER)")
    # plt.show()

# subplots()
pesq_plot()
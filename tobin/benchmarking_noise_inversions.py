"""This file will import from the standard library in tools.py where we having working rounding and noise functions. Run `test_embeddings_and_rounding` to ensure the functions are working as expected. 

Then we will run a series of tests here to see how the embeddings are affected by noise and how well the inversion works. The end result will be a publication ready figure. 

Date: Apr 24, 2024
Author: Tobin
"""
# %%

from tools import *
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import pandas as pd
import random
import csv
script_dir  = os.path.dirname(__file__)


# %% First we'll work with a single sentence to test and provide an example
text_to_embed = ['This is an example sentence for LLM embedding and reconstruction.']

# Let's establish a baseline
embeddings = embed_text(text_to_embed)
inverted = invert_embedding(embeddings)
inverted_embedding = embed_text(inverted)
baseline_cos_diff = torch.cosine_similarity(embeddings, inverted_embedding)

# %% Now we'll loop through and add some noise to the embeddings

embedding_range = [embeddings.min().item(), embeddings.max().item()]
cosine_diffs_pre_invert = {}
cosine_diffs_post_invert = {}
inversions = {}
diff_ranges = {}
n_steps = 6

for i in tqdm(range(n_steps)):
    edges, diff = get_edges(i, embedding_range)
    rounded = fakeround(edges, embeddings)
    gaussed = gauss_noise(embeddings, diff)
    uniformed = uniform_noise(embeddings, diff)

    diff_ranges[i] = diff

    cosine_diffs_pre_invert[i] = {'rounded': torch.cosine_similarity(embeddings, rounded),
                                  'gaussed': torch.cosine_similarity(embeddings, gaussed),
                                  'uniformed': torch.cosine_similarity(embeddings, uniformed)}
    
    inverted_rounded = invert_embedding(rounded)
    inverted_gaussed = invert_embedding(gaussed)
    inverted_uniformed = invert_embedding(uniformed)

    inversions[i] = {'rounded': inverted_rounded,
                  'gaussed': inverted_gaussed,
                  'uniformed': inverted_uniformed}
    

    inv_rounded_embedding = embed_text(inverted_rounded)
    inv_gaussed_embedding = embed_text(inverted_gaussed)
    inv_uniformed_embedding = embed_text(inverted_uniformed)

    cosine_diffs_post_invert[i] = {'rounded': torch.cosine_similarity(embeddings, inv_rounded_embedding),
                                    'gaussed': torch.cosine_similarity(embeddings, inv_gaussed_embedding),
                                    'uniformed': torch.cosine_similarity(embeddings, inv_uniformed_embedding)}
    

# %%Print results
print("Original text", text_to_embed)
print("Basic inversion", inverted)
print("Inversions for increasing noise")
for i in range(n_steps):
    print(f"Step {i}")
    print("Rounded:", inversions[i]['rounded'])
    print("Gaussed:", inversions[i]['gaussed'])
    print("Uniformed:", inversions[i]['uniformed'])


# %%Make the figure
x = [diff_ranges[i]/4 for i in range(n_steps)]
x = [_x/(embedding_range[1] - embedding_range[0]) for _x in x]
y_baseline = baseline_cos_diff.item()
y_rounded_pre = [cosine_diffs_pre_invert[i]['rounded'].item() for i in range(n_steps)]
y_gaussed_pre = [cosine_diffs_pre_invert[i]['gaussed'].item() for i in range(n_steps)]
y_uniformed_pre = [cosine_diffs_pre_invert[i]['uniformed'].item() for i in range(n_steps)]

y_rounded = [cosine_diffs_post_invert[i]['rounded'].item() for i in range(n_steps)]
y_gaussed = [cosine_diffs_post_invert[i]['gaussed'].item() for i in range(n_steps)]
y_uniformed = [cosine_diffs_post_invert[i]['uniformed'].item() for i in range(n_steps)]

fig, ax = plt.subplots()
ax.axhline(y_baseline, color='k', linestyle='dashdot', label='Baseline ')

ax.plot(x, y_rounded_pre, linestyle='--', alpha=0.5, c = 'r')
ax.plot(x, y_gaussed_pre, linestyle='--', alpha=0.5, c = 'g')
ax.plot(x, y_uniformed_pre,  linestyle='--', alpha=0.5, c = 'b')

ax.plot(x, y_rounded, label='Rounded', c = 'r')
ax.plot(x, y_gaussed, label='Gauss Noise', c = 'g')
ax.plot(x, y_uniformed, label='Uniform Noise', c = 'b')

ax.set_xlabel('Mean embedding noise (% of embedding range)')
ax.set_ylabel('Cosine similarity with original embedding')

ax.legend()
h,l = ax.get_legend_handles_labels()
custom_lines = [Line2D([0], [0], color='k', linestyle='--', alpha=0.5), Line2D([0], [0], linestyle='-', color='k', )]
ax.legend(h+custom_lines, l+['Pre Inversion (--)', 'Post Inversion (-)'], title="Modification type", loc='lower left')

fig.savefig( script_dir + '/results/SingleSentenceTest.png')
#fig.savefig('/u/tsouth/python/Python-3.10.6/python /u/tsouth/projects/PSI_SirMyce/tobin/results/SingleSentenceTest.png')




# %% Part 2. Now we'll work with multiple sentences to test and provide an example

# The goal here is to do the same as above but will a whole bunch of sentences and then average over the cosine similarities. 

#Lets get some more sentences to test
list_of_texts_to_embed = ['This is an example sentence for LLM embedding and reconstruction.',
                        'A second sentence to initially check everything works before running the loop 20 times']

#get a random sample of 100 sentences from the dataset
filename = script_dir + '/cv-unique-has-end-punct-sentences.csv'
n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
s = 10 #desired sample size
skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
df = pd.read_csv(filename, skiprows=skip)
long_list_of_texts_to_embed = df['sentence'].tolist() #turn sentences into list


#Our Baselines: (do this in a for loop for better memory)
embeddings = torch.empty((0,768))
inverted_embedding = torch.empty((0,768))
inverted = []
for text in long_list_of_texts_to_embed:
    embedding = embed_text([text])
    embeddings = torch.cat([embeddings,embedding], dim = 0)
    temp = invert_embedding(embedding)
    inverted.extend(temp)
for text in inverted:
    inverted_embedding_single = embed_text([text])
    inverted_embedding = torch.cat([inverted_embedding,inverted_embedding_single], dim = 0)

baseline_cos_diff = torch.cosine_similarity(embeddings, inverted_embedding)



# %% Now we'll loop through and add some noise to the embeddings

embedding_range = []
for r in range(0,len(long_list_of_texts_to_embed)):
    embedding_range.extend([[embeddings[r].min().item(), embeddings[r].max().item()]])
cosine_diffs_pre_invert = {}
cosine_diffs_post_invert = {}
inversions = {}
diff_ranges = {}
n_steps = 6

with open('sentence_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["original text", 'original inversion', 'inverted rounded text', 'inverted linRounded text','inverted gaussed text', 'inverted uniformed text', 
                     'cosine diff pre rounded', 'cosine diff pre linRounded','cosine diff pre gaussed', 'cosine diff pre uniformed', 
                     'cosine diff post rounded', 'cosine diff post linRounded', 'cosine diff post gaussed', 'cosine diff post uniformed'])
    for text in range(0,len(long_list_of_texts_to_embed)):
        diff_ranges[text] = {}
        cosine_diffs_pre_invert[text] = {}
        cosine_diffs_post_invert[text] = {}
        inversions[text] = {}
        for i in tqdm(range(n_steps)):
            edges, diff = get_edges(i, embedding_range[text])
            linspace_edges = get_linspace_edges(i, embedding_range[text])
        
            linRounded = linspace_round(linspace_edges, embeddings[text])
            rounded = fakeround(edges, embeddings[text])
            gaussed = gauss_noise(embeddings[text], diff)
            uniformed = uniform_noise(embeddings[text], diff)

            diff_ranges[text][i] = diff

            rounded = torch.transpose(rounded, 0, 1)
            linRounded = torch.transpose(linRounded, 0, 1)
            gaussed = gaussed.unsqueeze(0)
            uniformed = uniformed.unsqueeze(0)

            cosine_diffs_pre_invert[text][i] = {'rounded': torch.cosine_similarity(embeddings[text].unsqueeze(0), rounded),
                                    'linRounded': torch.cosine_similarity(embeddings[text].unsqueeze(0), linRounded),
                                    'gaussed': torch.cosine_similarity(embeddings[text].unsqueeze(0), gaussed),
                                    'uniformed': torch.cosine_similarity(embeddings[text].unsqueeze(0), uniformed)}

            inverted_rounded = invert_embedding(rounded)
            inverted_linRounded = invert_embedding(linRounded)
            inverted_gaussed = invert_embedding(gaussed)
            inverted_uniformed = invert_embedding(uniformed)

            inversions[text][i] = {'rounded': inverted_rounded,
                        'linRounded': inverted_linRounded,
                        'gaussed': inverted_gaussed,
                        'uniformed': inverted_uniformed}

            inv_rounded_embedding = embed_text(inverted_rounded)
            inv_linRounded_embedding = embed_text(inverted_linRounded)
            inv_gaussed_embedding = embed_text(inverted_gaussed)
            inv_uniformed_embedding = embed_text(inverted_uniformed)

            cosine_diffs_post_invert[text][i] = {'rounded': torch.cosine_similarity(embeddings[text].unsqueeze(0), inv_rounded_embedding),
                                            'linRounded': torch.cosine_similarity(embeddings[text].unsqueeze(0), inv_linRounded_embedding),
                                            'gaussed': torch.cosine_similarity(embeddings[text].unsqueeze(0), inv_gaussed_embedding),
                                            'uniformed': torch.cosine_similarity(embeddings[text].unsqueeze(0), inv_uniformed_embedding)}    
        
        #print the results
        print("Original text", long_list_of_texts_to_embed[text])
        print("Basic inversion", inverted[text])
        print("Inversions for increasing noise")
        for j in range(n_steps):
            print(f"Step {j}")
            print("Rounded:", inversions[text][j]['rounded'])
            print("LinRounded:", inversions[text][j]['linRounded'])
            print("Gaussed:", inversions[text][j]['gaussed'])
            print("Uniformed:", inversions[text][j]['uniformed'])
        #write the results to the csv file
        writer.writerow([long_list_of_texts_to_embed[text], inverted[text], inversions[text][j]['rounded'], inversions[text][j]['linRounded'],inversions[text][j]['gaussed'], inversions[text][j]['uniformed'], 
                        cosine_diffs_pre_invert[text][j]['rounded'].item(), cosine_diffs_pre_invert[text][j]['linRounded'].item(), cosine_diffs_pre_invert[text][j]['gaussed'].item(), cosine_diffs_pre_invert[text][j]['uniformed'].item(), 
                        cosine_diffs_post_invert[text][j]['rounded'].item(), cosine_diffs_post_invert[text][j]['linRounded'].item(), cosine_diffs_post_invert[text][j]['gaussed'].item(), cosine_diffs_post_invert[text][j]['uniformed'].item()])


# %%Make the figure with the Average results from the different texts
num_texts = len(long_list_of_texts_to_embed)

#function to compute mean and margin of error - key is either 'rounded', 'linRounded', 'gaussed', or 'uniformed'
def compute_mean_and_error(values, key, num_texts):
    means = [sum(values[text][i][key].item() for text in range(num_texts)) / num_texts for i in range(n_steps)]
    std_devs = [np.std([values[text][i][key].item() for text in range(num_texts)]) for i in range(n_steps)]
    std_errors = [std_dev / np.sqrt(num_texts) for std_dev in std_devs]
    margins_of_error = [1.96 * std_error for std_error in std_errors]
    return means, margins_of_error

x = [(sum(diff_ranges[text][i] for text in range(num_texts)) / num_texts) / 4 for i in range(n_steps)]
x = [_x/((sum(embedding_range[text][1] for text in range(num_texts)) / num_texts) - (sum(embedding_range[text][0] for text in range(num_texts)) / num_texts)) for _x in x]
y_baseline = (sum(baseline_cos_diff[text].item() for text in range(num_texts)) / num_texts)

y_rounded_pre, y_rounded_pre_err = compute_mean_and_error(cosine_diffs_pre_invert, 'rounded', num_texts)
y_linRounded_pre, y_linRounded_pre_err = compute_mean_and_error(cosine_diffs_pre_invert, 'linRounded', num_texts)
y_gaussed_pre, y_gaussed_pre_err = compute_mean_and_error(cosine_diffs_pre_invert, 'gaussed', num_texts)
y_uniformed_pre, y_uniformed_pre_err = compute_mean_and_error(cosine_diffs_pre_invert, 'uniformed', num_texts)

y_rounded, y_rounded_err = compute_mean_and_error(cosine_diffs_post_invert, 'rounded', num_texts)
y_linRounded, y_linRounded_err = compute_mean_and_error(cosine_diffs_post_invert, 'linRounded', num_texts)
y_gaussed, y_gaussed_err = compute_mean_and_error(cosine_diffs_post_invert, 'gaussed', num_texts)
y_uniformed, y_uniformed_err = compute_mean_and_error(cosine_diffs_post_invert, 'uniformed', num_texts)

#Plot Results
fig, ax = plt.subplots()
ax.axhline(y_baseline, color='k', linestyle='dashdot', label='Baseline ')

#Plot pre-inversion lines with error bars
ax.errorbar(x, y_rounded_pre, yerr=y_rounded_pre_err, linestyle='--', alpha=0.5, color='r', label='Rounded Pre')
ax.errorbar(x, y_linRounded_pre, yerr=y_linRounded_pre_err, linestyle='--', alpha=0.5, color='purple', label='LinRounded Pre')
ax.errorbar(x, y_gaussed_pre, yerr=y_gaussed_pre_err, linestyle='--', alpha=0.5, color='g', label='Gauss Noise Pre')
ax.errorbar(x, y_uniformed_pre, yerr=y_uniformed_pre_err, linestyle='--', alpha=0.5, color='b', label='Uniform Noise Pre')

#Plot post-inversion lines with error bars
ax.errorbar(x, y_rounded, yerr=y_rounded_err, color='r', label='Rounded')
ax.errorbar(x, y_linRounded, yerr=y_linRounded_err, color='purple', label='LinRounded')
ax.errorbar(x, y_gaussed, yerr=y_gaussed_err, color='g', label='Gauss Noise')
ax.errorbar(x, y_uniformed, yerr=y_uniformed_err, color='b', label='Uniform Noise')

ax.set_xlabel('Mean embedding noise (% of embedding range)')
ax.set_ylabel('Cosine similarity with original embedding')

ax.legend()
h,l = ax.get_legend_handles_labels()
custom_lines = [Line2D([0], [0], color='k', linestyle='--', alpha=0.5), Line2D([0], [0], linestyle='-', color='k', )]
ax.legend(h+custom_lines, l+['Pre Inversion (--)', 'Post Inversion (-)'], title="Modification type", loc='lower left')

fig.savefig( script_dir + '/results/10SentenceTest.png')
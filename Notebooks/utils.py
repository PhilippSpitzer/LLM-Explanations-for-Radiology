import os
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt 
import scipy.stats as stats
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLMResultsWrapper
import textwrap
import matplotlib.ticker as mtick
import re
import warnings


order_of_supplements = []

def round(num, ndec):
    import math
    multiplier = 10 ** ndec
    return math.floor(num * multiplier + 0.5) / multiplier


### mapping functions
def map_condition(i):
  if i == 1: return 'control'
  elif i == 2: return 'standard'
  elif i == 3: return 'chain-of-thought'
  elif i == 4: return 'differential'
  else: raise ValueError
 
def map_condition_label(label):
    label = label.strip()
    if label == 'Differential' or label == 'differential':
        return 'Differential diagnosis'
    return label
 
### plotting functions
def bar_annotate_n(sample_sizes,y=10,ax=None):
    if ax is None: ax = plt.gca()
    for bar, n in zip(ax.patches, sample_sizes):
        x = bar.get_x() + bar.get_width() / 2
        ax.text(x, y, f'$n={n}$', ha='center', va='bottom', fontsize=10, fontdict={'color': 'white'})
    
def format_ylab(ylab=None,ax=None): 
    if ax is None: ax = plt.gca()   
    if ylab is None: 
        l = ax.get_ylabel().replace('-',' ').capitalize()
        ax.set_ylabel(l)
    else: ax.set_ylabel(ylab)
    
def format_xlab(xlab=None,ax=None):
    if ax is None: ax = plt.gca()
    if xlab is None: 
        l = ' '.join(word.capitalize() for word in ax.get_xlabel().split())
        ax.set_xlabel(l)
    else: ax.set_xlabel(xlab)
    
def capitalize_xticklabels(ax=None): 
    if ax is None: ax = plt.gca()
    ax.set_xticklabels([_.get_text().capitalize() for _ in ax.get_xticklabels()])
def wrap_xticklabels(labelwrap,ax=None):
    if ax is None: ax = plt.gca()
    labels = [textwrap.fill(map_condition_label(tick.get_text()), width=labelwrap) for tick in plt.gca().get_xticklabels()]
    ax.set_xticklabels(labels, rotation=0)

def format_percentage(perc,ax=None): 
    if ax is None: ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=perc))

def format_labs(ylab=None,xlab=None,ylim=(0,100),capitalize=True,perc=100,labelwrap=12,ax=None):
    # careful with interactions btw ylim and perc
    if ax is None: ax = plt.gca()
    format_xlab(xlab); format_ylab(ylab)
    if capitalize: capitalize_xticklabels()
    if ylim is not None: ax.set_ylim(ylim)
    if perc is not None: format_percentage(perc)
    if labelwrap is not None: wrap_xticklabels(labelwrap)

def add_grid(ax=None):
    if ax is None: ax = plt.gca()
    ax.grid(visible=True,which='major',axis='x')
    
def save_plot(name,ax=None): 
    if ax is None: ax = plt.gca()
    try:
        ax.get_figure().savefig(f'../Results/Plots/{name}.pdf')
    except FileNotFoundError:
        warnings.warn('Create "../Results/Plots folder" to save figure files')
 
def annotate_tests(p_values,order,ymax,ax=None, low_test_margin=0.04,high_test_margin=0.015,low_offset=0):
    if ax is None: ax = plt.gca()
    # Add statistical annotations (e.g., p-values or significance stars)
    for i, ((group1, group2), (t_val, p_val)) in enumerate(p_values.items()):
        # Choose the positions for the annotations 
        x1 = order.index(group1)
        x2 = order.index(group2)
        y_max = ymax - 0.0 + i * 0.07  # some padding
        
        # Annotate with a significance star based on the p-value
        if p_val < 0.001: annotation = '***'
        elif p_val < 0.01: annotation = '**'
        elif p_val < 0.05: annotation = '*'
        else: annotation = ''  # Not significant
        
        # Add the annotation to the plot
        p_val_annotated = f'$P = {p_val:.3f}${annotation}' if p_val >= 0.001 else f'$P < 0.001${annotation}'
        if i < 3:
            ax.plot([x1, x1, x2, x2], [y_max, y_max + 0.01,y_max+ 0.01, y_max], linewidth=0.5,color='black', clip_on=False)
            ax.text((x1 + x2) / 2, y_max + high_test_margin, p_val_annotated, ha='center', va='bottom', fontsize=8)
        else: 
            y_max -= 1.02 + low_offset
            ax.plot([x1, x1, x2, x2], [y_max+ 0.01, y_max,y_max, y_max+ 0.01], linewidth=0.5,color='black', clip_on=False)
            ax.text((x1 + x2) / 2, y_max - low_test_margin, p_val_annotated, ha='center', va='bottom', fontsize=8)
            
# Perform independent one-sided t-tests
def one_sided_ttest(group1, group2):
  t_statistic, p_value = stats.ttest_ind(group1, group2, alternative='less', equal_var=False)
  return t_statistic, p_value

def fit(df,target,features,model=sm.OLS):
    if not isinstance(features, list): features = [features]
    X = pd.get_dummies(df[features], drop_first=True)
    X = sm.add_constant(X).astype(float)
    Y = df[target]
    model = model(Y, X).fit()
    return model

### get data
def prepare_adh_df(df,map_values):
    df_adh_map = pd.read_excel('../Data/adherence_summary.xlsx',).rename({'answer_participant':'answer'},axis=1)
    df_adh = df[~(df.condition=='control')]
    df_adh = pd.merge(df_adh.reset_index(),df_adh_map,how='left',on=['question','answer','condition'])
    df_adh['Adherence'] = df_adh['Adherence'].replace(map_values).map({'yes':1,'no':0})
    return df_adh.groupby('ResponseId').agg({'condition': 'first', 'Adherence': 'mean',})





def get_tabular(model=None,summary_df=None, const_name="Intercept", replacements={'Ai':'AI','It':'IT'},
                index_formatter=None, n_dc = 3, column_format='lrrrr',
                add_info={'AIC':'aic','Obs. ($N$)': 'nobs'}):

    if summary_df is None and model is not None:
    # Extract coefficients, standard errors, p-values, and confidence intervals
        summary_df = pd.DataFrame({
            "Coef.": model.params,
            "s.e.": model.bse,
            "$P$-value": [p_val if p_val >= 0.0005 else '$< 0.001$' for p_val in model.pvalues],
            "95 % CI": [f"[{low:.3f}; {high:.3f}]" for low, high in zip(model.conf_int()[0], model.conf_int()[1])]
        })

        summary_df.index = model.params.index #["Intercept", "Self-Save", "Self-Earn", "Environment-CO2"]
    elif summary_df is not None and model is None:
        summary_df = summary_df
    else: raise ValueError(f"Supply either model or summary_df!")
    
    summary_df = summary_df.rename(index={'const': const_name},)
    
    if index_formatter is None: 
        index_formatter = lambda o: '\\textit{' + o.replace('_',': ').title() + '}'
    summary_df = summary_df.rename(index=index_formatter)

    # Print as LaTeX-style table
    tabular = summary_df.to_latex(float_format=f"%.{n_dc}f",escape=False,column_format=column_format,longtable=False)
    tabular = tabular.replace('\\bottomrule','\midrule')
    tabular = tabular.replace('\end{tabular}','')
    tabular = tabular.replace('%','\%')
    for k,v in replacements.items():
        tabular = tabular.replace(k,v)
    
    # Additional model information
    from operator import attrgetter
    if isinstance(add_info,dict):
        info = ''
        for k,v in add_info.items():
            getter = attrgetter(str(v))    
            try: o = getter(model)
            except AttributeError: o = v
            if k == 'Obs. ($N$)':info += f"{k} & & & & {int(o)} \\\\ \n"
            else: info += f"{k} & & & & {o:.3f} \\\\ \n"
        tabular += info

    tabular += '\\bottomrule\n\end{tabular}'
    tabular = re.sub(r'-(?=\d)', r'$-$', tabular)
    return tabular

def to_table(tabular, fn, caption_text=None, label_text=None, center=True, rowwidth=1, footnotesize=True):
    sum = "\\begin{table}\n"
    if center: sum += "\\begin{center}\n"
    caption_text = caption_text if caption_text is not None else fn
    caption_text = latex_minus_and_p(caption_text)
    if footnotesize: sum += "\\begingroup\n\\footnotesize\n"
    if rowwidth is not None: sum += f"\\renewcommand{{\\arraystretch}}{{{rowwidth}}}\n"
    sum += tabular if not isinstance(tabular, list) else "".join(tabular)
    if footnotesize: sum += "\\endgroup\n"
    if footnotesize: caption_text = "\\footnotesize " + caption_text
    sum += f'\caption{{{caption_text}}}' 
    sum += f'\n\label{{{label_text}}}' if label_text is not None else f'\n\label{{{"tab:"+fn.replace(" ", "-")}}}'
    if center: sum += "\n\\end{center}"
    sum += '\n\end{table}'
    return sum

def save_tex(fn,model=None,summary_df=None,caption=None, label=None,center=True,rowwidth=1,footnotesize=True,**kwargs):
    tabular = get_tabular(model=model,summary_df=summary_df,**kwargs)
    table = to_table(tabular,fn,caption,label,center,rowwidth,footnotesize)
    order_of_supplements.append(fn)
    try:
        with open(f"../Results/Tex/{fn}.tex", 'w') as f: 
            f.write(table)
    except FileNotFoundError:
        warnings.warn('Create "../Results/Tex folder" to save tex files')
        
# Save a dataframe to tex file
def save_tabtex(o, fn, cap='Caption', lab='tab:my_label',escape=True, n_dec=3,footnotesize=True,rowwidth=1):
    tex = '\\begin{table}\n\centering\n'
    if footnotesize: tex += "\\begingroup\n\\footnotesize\n"
    if rowwidth is not None: tex += f"\\renewcommand{{\\arraystretch}}{{{rowwidth}}}\n"
    # if fontsize is not None: tex += fontsize + '\n'
    tex += o.to_latex(escape=escape, float_format=f"%.{n_dec}f")
    if footnotesize: 
        tex += "\\endgroup\n"
        caption_text = f'\caption{{\\footnotesize {cap}}}\n' 
    else: caption_text = f'\caption{{{cap}}}\n'
    tex += caption_text
    tex += f'\label{{{lab}}}\n'
    tex += '\\end{table}'
    
    try:
        with open(f"../Results/Tex/{fn}.tex", "w") as f:
            f.write(tex)
    except FileNotFoundError:
        warnings.warn('Create "../Results/Tex folder" to save tex files')
    order_of_supplements.append(fn)


# def save_mixedlm(fn, model, caption=None, label=None, fontsize='footnotesize'):
    
#     if caption is not None: caption = f'\caption{{{caption}}}'
#     else: caption = f'\caption{{{fn}}}'
#     if label is not None: caption += f'\n\label{{{label}}}'
#     else:
#         label = fn.replace(' ','_') 
#         caption += f'\n\label{{tab:{label}}}'
            
#     sum = model.summary().as_latex()
#     sum = sum.replace('\n\caption{Mixed Linear Model Regression Results}', '')
#     sum = sum.replace('\n\label{}', '')
#     sum = sum.replace('\\bigskip', '')
#     sum = sum.strip()
#     sum = sum.split('\end{table}')
#     sum = sum[0] + caption + sum[1] + '\n\end{table}'
#     with open(f"../Results/Tex/{fn}.tex", "w") as f:
#         f.write(sum)
    
        
        
def escape_percent(s):
    return re.sub(r'(?<!\\)%', r'\%', s)

def latex_minus_and_p(s):
    s = re.sub(r'-(?=\d)', r'$-$', s)
    s = re.sub(r"P-", r"$P$-", s)
    # s = re.sub(r"P <", r"$P$ <", s)
    # s = re.sub(r"P >", r"$P$ >", s)
    # pattern = r"(\[[^$]*?)-(\d+)([^$]*?\])"
    # replacement = r"\1$-$\2\3"
    # s = re.sub(pattern, replacement, s)
    return s
    
# consolidate tex files
def consolidate_tex(add_header=False):
    o = ''
    fn = "supplementary_materials.tex"
    #for f in reversed(sorted(os.listdir('../Results/Tex',))):
    for f in order_of_supplements:
        try:
            f = f + '.tex'
            print(f)
            if f == fn: continue
            if add_header:
                o += f"\section*{{{f.replace('.tex','')}}}\n"
                o += f"\label{{sec:{f.replace(' ','_')}}}\n"
                    
            with open(f"../Results/Tex/{f}", "r") as f: 
                o += escape_percent(f.read())
            o += '\n\n'
        except FileNotFoundError:
            warnings.warn('Create "../Results/Tex folder" to save tex files')
    
    try:
        with open(f"../Results/Tex/{fn}", "w") as f:    
            f.write(o)
    except FileNotFoundError:
        warnings.warn('Create "../Results/Tex folder" to consolidatae tex files')
    
        
def get_gpt_review_df():
       output_review = pd.read_excel('../Data/LLM_output_reviews.xlsx',)
       rename_map = {'Correct Diagnosis Chain of Thought Reasoning': 'chain-of-thought diagnosis correct',
                     'Correct Diagnosis Standard': 'standard diagnosis correct', 
                     'Correct Diagnosis Differential Diagnosis': 'differential diagnosis correct',
                     'DD-Expl. Correct?': 'differential explanation correct',
                     'CoT-Expl. Correct?': 'chain-of-thought explanation correct',
                     'Std.-Expl. Correct?': 'standard explanation correct',}
       output_review.rename(rename_map, inplace=True, axis=1)
       output_review = output_review[output_review['Include Case in Study'] == 'Yes']
       output_review.drop(['Spalte 30', 'Spalte 31', 'Spalte 32',
              'Spalte 33', 'Spalte 34', 'Spalte 35', 'Spalte 36', 'Spalte 37',
              'Spalte 38', 'Spalte 39', 'Spalte 40', 'Spalte 41', 'Spalte 42',
              'Spalte 43', 'Spalte 44'], inplace=True, axis=1)
       return output_review

# 
TASKS = {
    'question_1': "A previously healthy 5-year-old boy was brought to the surgery clinic with a 2-day history of intermittent abdominal pain. On palpation of the abdomen there was pain in the periumbilical region, but no rebound or guarding. An ultrasound was normal, and a computed tomography of the abdomen was performed (Panels A,B).",
    'question_2': "A 55-year-old man presented with 10 years of progressive handwriting impairment and rapid, slurred speech. In his thirties, he had worked as a welder without access to personal protective equipment. Neurologic examination was notable for reduced facial expression, blepharospasm, and cluttered, dysarthric speech. Postural reflexes were mildly impaired. MRI imaging of the head showed a nonenhancing, T1-weighted, hyperintense signal in the basal ganglia on both sides. Ceruloplasmin and iron levels were normal.",
    'question_3': "26-year-old man from Somalia presented with a 5-month history of dry cough, night sweats, and unintentional weight loss of 18 kg. During this period, epigastric pain and postprandial vomiting had also developed. His BMI was 11. On examination, he was cachectic with abdominal distention and diffuse tenderness to palpation. On the basis of chest imaging and sputum studies, a diagnosis of pulmonary tuberculosis was made, and intravenous antituberculous treatment was initiated. However, he continued to have postprandial vomiting. Contrast-enhanced CT of the abdomen was obtained.",
    'question_4': "A 35-year-old man with IgA nephropathy presented with confusion, blurry vision, and seizures. Two weeks before presentation, he had started receiving cyclosporine. Physical examination was notable for a blood pressure of 160/80 mm Hg, drowsiness, and decreased visual acuity. A fundoscopic examinations was normal. T2-weighted magnetic resonance imaging (MRI) with fluid-attenuated inversion recovery sequencing of the head was performed.",
    'question_5': "A 52-year-old woman with end-stage kidney disease that was being managed with peritoneal dialysis presented with a 1-month history of bloody dialysate. She had had 3 episodes of bacterial peritonitis in the past 12 years. Physical examination and laboratory studies were unremarkable. Computed tomography of the abdomen was performed.",
    'question_6': "A 32-year-old man presented with a 6-week history of tingling in his arms and legs and a 2-week history of inability to walk. A positive Romberg test, sensory ataxia, impaired proprioception and vibratory sensation, and preserved nociception were noted. Magnetic resonance imaging of the whole spine showed hyperintensity in the posterior spinal cord from C1 to T12 and hyperintense lesions in the dorsal column on T2-weighted images. A vitamin B12 level was 107 pg per ml (reference value, >231) without macrocytic anemia.",
    'question_7': "A 35-year-old woman with idiopathic pulmonary arterial hypertension and a pulmonary aneurysm presented with chest pain. Computed tomography (CT) of the chest is shown.",
    'question_8': "A 38-year-old man presented to the otolaryngology clinic with chronic difficulty breathing through his right nostril. Physical examination showed nasal septal deviation, calcified septal spurs, and a 2-cm perforation in the posterior septum. On rhinoscopy, a hard, nontender, white mass was observed in the floor of the right nostril. CT of the paranasal sinuses showed a well-defined, radiodense mass.",
    'question_9': "A 16-day-old girl was brought to the emergency department with lethargy. Physical exam showed tachypnea and marked hepatomegaly, as well as small hemangiomas on the skin. TSH was elevated. MRI showed numerous hepatic lesions and cardiomegaly.",
    'question_10': "A 71-year-old man was hospitalized with altered mental status progressing over the preceding 3 weeks. The patient had a recent diagnosis of adenocarcinoma of the colon with known metastatic lesions in the lung and bones. A gadolinium-enhanced magnetic resonance image of the brain was performed and is shown.",
    'question_11': "A 29-year-old man with perinatally acquired human immunodeficiency virus (HIV) infection and intermittent adherence to antiretroviral therapy presented to the hospital with abdominal pain and drenching night sweats. On presentation, his CD4 count was 18 cells per cubic millimeter (reference range, 500 to 1500), and the HIV viral load was undetectable. Physical exam showed severe abdominal distention, splenomegaly, and diffuse abdominal tenderness to palpation. Computed tomography of the abdomen confirmed massive splenomegaly with multifocal infarction of the splenic parenchyma.",
    'question_12': "A 42-year-old man presented to the clinic with a 3-month history of worsening cough, shortness of breath, and fever. Physical examination showed inflamed nasal mucosa and nasal crusting. Wheezes and rales were heard on auscultation. A computed tomographic scan of the face showed extensive destruction of the structural bones of the midface, resulting in a large nasal cavity. ",
    'question_13': "A 63-year-old man presented to the emergency department with a 3-day history of abdominal pain that had started in the periumbilical area and subsequently shifted to the left lower quadrant. Initial laboratory tests showed a white-cell count of 12,000 per cubic millimeter (reference range, 4000 to 10,000) and a lactate level of 1.8 mmol per liter (normal value, <1.9). Contrast-enhanced computed tomography of the abdomen revealed edema of the sigmoid colon with thumbprinting.",
    'question_14': "A 28-year-old woman with vertigo, confusion, and falls 2 weeks after a surgical abortion at 11 weeks of gestation presents to the emergency department. Examination revealed spontaneous upbeat nystagmus, gaze-evoked nystagmus, and gait ataxia.",
    'question_15': "A 59-year-old previously healthy man presented with progressively worsening headaches and bluish nodular skin lesions. Fast-field echo MRI image of the brain showed this finding.",
    'question_16': "A 44-year-old woman presented to the emergency department with acute chest pain after several months of progressive dyspnea. Her oxygen saturation was 92%, and she had diminished breath sounds on the right side. Chest CT revealed a large right-sided pneumothorax and diffuse, intraparenchymal pulmonary cysts. ",
    'question_17': "A 54-year-old man presented with a 3-week history of cognitive deterioration. Neurologic examination revealed disorientation, horizontal gaze-evoked nystagmus, hyperreflexia, startle myoclonus, and ataxia. Brain MRI with diffusion-weighted imaging revealed hyperintensity of the cortical gyri and caudate heads.",
    'question_18': "A 30-year-old man presented with a 15-month history of intermittent discomfort in the right upper quadrant of the abdomen. He lived in a rural area of Morocco and had occasional contact with dogs. The physical examination revealed hepatomegaly with a palpable hepatic mass. Laboratory tests showed a normal white-cell count and a normal absolute eosinophil count. Ultrasonography and computed tomography of the abdomen revealed a large cyst in the right lobe of the liver.",
    'question_19': "A 59-year-old woman presented to the emergency department with a 4-day history of inflammation and pain in the right eye. She had been blind in the eye for several years before presentation. Magnetic resonance imaging revealed a right orbital mass. Abdominal and thoracic imaging showed numerous hepatic masses, abdominal and thoracic lymphadenopathy, and vertebral sclerotic osseous disease. The right eye was enucleated for palliative relief and to obtain tissue for diagnosis.",
    'question_20': "An 18-year-old man presented to the emergency department with generalized tonic–clonic seizures. On physical examination, the patient was confused. He had swelling over the right eye and tenderness in the right testis. Magnetic resonance imaging of the head showed numerous well-defined cystic lesions throughout the cerebral cortex.",
}

GROUND_TRUTHS = {
    'question_1': 'Colocolonic intussusception',
    'question_2': 'manganese poisoning',
    'question_3': 'Superior mesenteric artery syndrome',
    'question_4': 'posterior reversible encephalopathy syndrome',
    'question_5': 'Encapsulating peritoneal sclerosis',
    'question_6': 'subacute combined degeneration',
    'question_7': 'Pulmonary-artery dissection',
    'question_8': 'inverted ectopic tooth',
    'question_9': 'Infantile hepatic hemangiomas',
    'question_10': 'Metastatic adenocarcinoma',
    'question_11': 'Disseminated Mycobacterium avium–intracellulare infection',
    'question_12': 'Granulomatosis with polyangiitis',
    'question_13': 'Ischemic colitis',
    'question_14': 'Wernicke’s encephalopathy',
    'question_15': 'Cerebral cavernous malformations',
    'question_16': 'Lymphangioleiomyomatosis',
    'question_17': 'Creutzfeld-Jakob disease',
    'question_18': 'Cystic echinococcosis',
    'question_19': 'Uveal melanoma',
    'question_20': 'Neurocysticercosis',
    }

PROMPT = """### Context: I posed a medical question to a radiologist. 
### Medical question: {medical_question}
### Ground truth: {ground_truth}.
### Radiologist's response: {response}.

### Instruction: Help me judge if the radiologist's response is correct.
Ignore spelling errors and small deviations from the ground truth.
Focos on the sematics: it is ok if the radiologist uses different words if their meaning is similar.
Accept common abbreviations of diseases as correct answers. Ignore difference due to signular/plural forms.
Only answer 'Yes' or 'No'."""


DIAGNOSES_GPT = [
    {'question': 'question_1', 'condition': 'differential', 'diagnosis': 'Intussusception'},
    {'question': 'question_2', 'condition': 'differential', 'diagnosis': 'Manganese Toxicity'},
    {'question': 'question_3', 'condition': 'differential', 'diagnosis': 'Abdominal Tuberculosis'},
    {'question': 'question_4', 'condition': 'differential', 'diagnosis': 'Posterior Reversible Encephalopathy Syndrome (PRES)'},
    {'question': 'question_5', 'condition': 'differential', 'diagnosis': 'Encapsulating Peritoneal Sclerosis (EPS)'},
    {'question': 'question_6', 'condition': 'differential', 'diagnosis': 'Subacute Combined Degeneration (SCD)'},
    {'question': 'question_7', 'condition': 'differential', 'diagnosis': 'Pulmonary Artery Aneurysm Rupture'},
    {'question': 'question_8', 'condition': 'differential', 'diagnosis': 'Rhinolith'},
    {'question': 'question_9', 'condition': 'differential', 'diagnosis': 'Infantile Hepatic Hemangioendothelioma (IHHE)'},
    {'question': 'question_10', 'condition': 'differential', 'diagnosis': 'Cerebral Metastases'},
    {'question': 'question_11', 'condition': 'differential', 'diagnosis': 'AIDS-related Lymphoma'},
    {'question': 'question_12', 'condition': 'differential', 'diagnosis': "Granulomatosis with Polyangiitis (Wegener's Granulomatosis)"},
    {'question': 'question_13', 'condition': 'differential', 'diagnosis': 'Diverticulitis'},
    {'question': 'question_14', 'condition': 'differential', 'diagnosis': "Wernicke's encephalopathy"},
    {'question': 'question_15', 'condition': 'differential', 'diagnosis': 'Metastatic melanoma'},
    {'question': 'question_16', 'condition': 'differential', 'diagnosis': 'Lymphangioleiomyomatosis (LAM)'},
    {'question': 'question_17', 'condition': 'differential', 'diagnosis': 'Creutzfeldt-Jakob disease (CJD)'},
    {'question': 'question_18', 'condition': 'differential', 'diagnosis': 'Hydatid disease (Echinococcosis)'},
    {'question': 'question_19', 'condition': 'differential', 'diagnosis': 'Metastatic carcinoma'},
    {'question': 'question_20', 'condition': 'differential', 'diagnosis': 'Neurocysticercosis'},
    {'question': 'question_1', 'condition': 'chain-of-thought', 'diagnosis': 'intussusception'},
    {'question': 'question_2', 'condition': 'chain-of-thought', 'diagnosis': 'Manganese Toxicity'},
    {'question': 'question_3', 'condition': 'chain-of-thought', 'diagnosis': 'Gastrointestinal Tuberculosis with involvement of the stomach or duodenum leading to gastric outlet obstruction'},
    {'question': 'question_4', 'condition': 'chain-of-thought', 'diagnosis': 'Posterior Reversible Encephalopathy Syndrome (PRES)'},
    {'question': 'question_5', 'condition': 'chain-of-thought', 'diagnosis': 'encapsulating peritoneal sclerosis (EPS)'},
    {'question': 'question_6', 'condition': 'chain-of-thought', 'diagnosis': 'subacute combined degeneration (SCD) of the spinal cord due to vitamin B12 deficiency'},
    {'question': 'question_7', 'condition': 'chain-of-thought', 'diagnosis': 'pulmonary artery dissection'},
    {'question': 'question_8', 'condition': 'chain-of-thought', 'diagnosis': 'rhinolith'},
    {'question': 'question_9', 'condition': 'chain-of-thought', 'diagnosis': 'congenital infantile hepatic hemangioma with associated hypothyroidism'},
    {'question': 'question_10', 'condition': 'chain-of-thought', 'diagnosis': 'brain metastases secondary to metastatic colon adenocarcinoma'},
    {'question': 'question_11', 'condition': 'chain-of-thought', 'diagnosis': 'disseminated Mycobacterium avium complex (MAC) infection'},
    {'question': 'question_12', 'condition': 'chain-of-thought', 'diagnosis': 'granulomatosis with polyangiitis (GPA)'},
    {'question': 'question_13', 'condition': 'chain-of-thought', 'diagnosis': 'acute diverticulitis'},
    {'question': 'question_14', 'condition': 'chain-of-thought', 'diagnosis': 'Wernicke encephalopathy'},
    {'question': 'question_15', 'condition': 'chain-of-thought', 'diagnosis': 'Cerebral Cavernous Malformation Syndrome (CCM Syndrome)'},
    {'question': 'question_16', 'condition': 'chain-of-thought', 'diagnosis': 'lymphangioleiomyomatosis (LAM)'},
    {'question': 'question_17', 'condition': 'chain-of-thought', 'diagnosis': 'Creutzfeldt-Jakob disease (CJD)'},
    {'question': 'question_18', 'condition': 'chain-of-thought', 'diagnosis': 'hepatic echinococcosis (hydatid disease)'},
    {'question': 'question_19', 'condition': 'chain-of-thought', 'diagnosis': 'metastatic cancer, specifically metastatic ocular melanoma'},
    {'question': 'question_20', 'condition': 'chain-of-thought', 'diagnosis': 'neurocysticercosis'},
    {'question': 'question_1', 'condition': 'standard', 'diagnosis': 'intussusception'},
    {'question': 'question_2', 'condition': 'standard', 'diagnosis': 'manganism'},
    {'question': 'question_3', 'condition': 'standard', 'diagnosis': 'tuberculous enteritis'},
    {'question': 'question_4', 'condition': 'standard', 'diagnosis': 'posterior reversible encephalopathy syndrome (PRES)'},
    {'question': 'question_5', 'condition': 'standard', 'diagnosis': 'encapsulating peritoneal sclerosis (EPS)'},
    {'question': 'question_6', 'condition': 'standard', 'diagnosis': 'subacute combined degeneration (SCD) of the spinal cord'},
    {'question': 'question_7', 'condition': 'standard', 'diagnosis': 'pulmonary aneurysm'},
    {'question': 'question_8', 'condition': 'standard', 'diagnosis': 'rhinolith'},
    {'question': 'question_9', 'condition': 'standard', 'diagnosis': 'neonatal hemangiomatosis'},
    {'question': 'question_10', 'condition': 'standard', 'diagnosis': 'metastatic lesions to the brain'},
    {'question': 'question_11', 'condition': 'standard', 'diagnosis': 'HIV-associated lymphoma'},
    {'question': 'question_12', 'condition': 'standard', 'diagnosis': 'granulomatosis with polyangiitis (GPA)'},
    {'question': 'question_13', 'condition': 'standard', 'diagnosis': 'acute diverticulitis'},
    {'question': 'question_14', 'condition': 'standard', 'diagnosis': 'Wernicke encephalopathy'},
    {'question': 'question_15', 'condition': 'standard', 'diagnosis': 'cerebral cavernous malformations (CCMs)'},
    {'question': 'question_16', 'condition': 'standard', 'diagnosis': 'lymphangioleiomyomatosis (LAM)'},
    {'question': 'question_17', 'condition': 'standard', 'diagnosis': 'Creutzfeldt-Jakob disease (CJD)'},
    {'question': 'question_18', 'condition': 'standard', 'diagnosis': 'hepatic hydatid cyst'},
    {'question': 'question_19', 'condition': 'standard', 'diagnosis': 'intraocular tumor'},
    {'question': 'question_20', 'condition': 'standard', 'diagnosis': 'neurocysticercosis'},
]

DD_GPT = [
    {"question": "question_1", "condition": "differential", "diagnosis": ["Mesenteric Adenitis", "Acute Gastroenteritis", "Appendicitis", "Constipation"]}, 
    {"question": "question_2", "condition": "differential", "diagnosis": ["Parkinson's Disease", "Wilson's Disease", "Progressive Supranuclear Palsy (PSP)", "Multiple System Atrophy (MSA)"]}, 
    {"question": "question_3", "condition": "differential", "diagnosis": ["Superior Mesenteric Artery (SMA) Syndrome", "Gastric Outlet Obstruction", "Malignancy (e.g., Gastric Cancer)", "Chronic Pancreatitis"]}, 
    {"question": "question_4", "condition": "differential", "diagnosis": ["Cyclosporine Neurotoxicity", "Hypertensive Encephalopathy", "Central Nervous System Infection", "Metabolic Encephalopathy"]}, 
    {"question": "question_5", "condition": "differential", "diagnosis": ["Peritoneal Dialysis-Associated Peritonitis", "Peritoneal Carcinomatosis", "Mesenteric Ischemia", "Hemorrhagic Cyst of the Ovary"]}, 
    {"question": "question_6", "condition": "differential", "diagnosis": ["Multiple Sclerosis (MS)", "Neurosyphilis", "Spinal Cord Tumor", "HIV-Associated Myelopathy"]}, 
    {"question": "question_7", "condition": "differential", "diagnosis": ["Pulmonary Embolism", "Aortic Dissection", "Myocardial Infarction", "Pneumothorax"]}, 
    {"question": "question_8", "condition": "differential", "diagnosis": ["Osteoma", "Inverted Papilloma", "Foreign Body", "Nasal Polyp with Calcification"]}, 
    {"question": "question_9", "condition": "differential", "diagnosis": ["Congenital Hypothyroidism", "Neonatal Hemochromatosis", "Congenital Heart Disease", "TORCH Infection (e.g., Cytomegalovirus)"]}, 
    {"question": "question_10", "condition": "differential", "diagnosis": ["Primary CNS Lymphoma", "Brain Abscesses", "Progressive Multifocal Leukoencephalopathy (PML)", "Glioblastoma Multiforme"]}, 
    {"question": "question_11", "condition": "differential", "diagnosis": ["Disseminated Mycobacterium avium Complex (MAC) Infection", "Splenic Abscess", "Cytomegalovirus (CMV) Infection", "Leishmaniasis"]}, 
    {"question": "question_12", "condition": "differential", "diagnosis": ["Invasive Fungal Sinusitis", "Extranodal NK/T-cell Lymphoma, Nasal Type", "Chronic Rhinosinusitis with Nasal Polyposis", "Sarcoidosis"]}, 
    {"question": "question_13", "condition": "differential", "diagnosis": ["Ischemic colitis", "Infectious colitis", "Inflammatory bowel disease (IBD)", "Colonic neoplasm"]}, 
    {"question": "question_14", "condition": "differential", "diagnosis": ["Cerebellar stroke", "Multiple sclerosis (MS)", "Central nervous system (CNS) infection", "Brain tumor"]}, 
    {"question": "question_15", "condition": "differential", "diagnosis": ["Primary brain tumor", "Cerebral amyloid angiopathy", "Systemic lupus erythematosus (SLE) with CNS involvement", "Kaposi's sarcoma"]}, 
    {"question": "question_16", "condition": "differential", "diagnosis": ["Chronic obstructive pulmonary disease (COPD)", "Pulmonary Langerhans cell histiocytosis (PLCH)", "Bullous emphysema", "Pneumocystis jirovecii pneumonia (PCP)"]}, 
    {"question": "question_17", "condition": "differential", "diagnosis": ["Herpes simplex encephalitis (HSE)", "Wernicke encephalopathy", "Autoimmune encephalitis", "Progressive multifocal leukoencephalopathy (PML)"]}, 
    {"question": "question_18", "condition": "differential", "diagnosis": ["Hepatic cystadenoma", "Pyogenic liver abscess", "Hepatic hemangioma", "Hepatocellular carcinoma (HCC)"]}, 
    {"question": "question_19", "condition": "differential", "diagnosis": ["Orbital lymphoma", "Orbital pseudotumor (Idiopathic Orbital Inflammatory Disease)", "Orbital sarcoma", "Sarcoidosis"]}, 
    {"question": "question_20", "condition": "differential", "diagnosis": ["Toxoplasmosis", "Tuberculosis", "Metastatic disease", "Sarcoidosis"]}
    ]
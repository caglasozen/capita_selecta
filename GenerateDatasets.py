"""
@Author: Thomas Boot
@Author: Cagla Sozen
@Date: 21/02/2021
"""

from river import synth
import pandas as pd

dataset_1 = synth.ConceptDriftStream(stream=synth.SEA(seed=42, variant=3), drift_stream=synth.SEA(seed=42, variant=2), seed=1, position=250000, width=1)
# Abrupt drift using the SEA dataset. Currently set for high abrupt drift. Adjust parameter settings as needed.
SEA_abrupt = dataset_1.take(500000)
frame = []
series = []
for x,y in SEA_abrupt:
    frame.append(x)
    series.append(y)

df1 = pd.DataFrame(frame)
df1['y']=series
df1.to_csv('data\SEA_Abrupt_high.csv')

dataset_2 = synth.Hyperplane(seed=42,n_drift_features=10,mag_change=0.1,noise_percentage=0,sigma=0.1)
# Gradual drift using the Hyperplane dataset. Currently set for high gradual drift. Adjust parameter settings as needed.
Hyperplane_grad_01 = dataset_2.take(500000)
frame=[]
series=[]
for x,y in Hyperplane_grad_01:
    frame.append(x)
    series.append(y)

df2 = pd.DataFrame(frame)
df2['y']=series
df2.to_csv('data\Hyperplane_01_high.csv')

dataset_3 = synth.ConceptDriftStream(stream=synth.Agrawal(seed=42), drift_stream=synth.ConceptDriftStream(stream=synth.Agrawal(classification_function=2,seed=42),
drift_stream=synth.ConceptDriftStream(stream=synth.Agrawal(seed=42),drift_stream=synth.Agrawal(classification_function=4,seed=42),seed=1,position=250000,width=100000),seed=1,position=250000,width=1), seed=1, position=250000, width=100000)
# mixed drift using the Agrawal dataset. 2 gradual drifts (before and after 500k) and 1 abrupt drift (at 500k).
Agrawal_mixed = dataset_3.take(1000000)
frame=[]
series=[]
for x,y in Agrawal_mixed:
    frame.append(x)
    series.append(y)

df3 = pd.DataFrame(frame)
df3['y']=series
df3.to_csv('data\Agrawal_mixed.csv')

dataset_4 = synth.ConceptDriftStream(stream=synth.SEA(seed=42, variant=3), drift_stream=synth.SEA(seed=42, variant=2), seed=1, position=250000, width=100)
SEA_grad = dataset_4.take(500000)
frame = []
series = []
for x,y in SEA_grad:
    frame.append(x)
    series.append(y)

df4 = pd.DataFrame(frame)
df4['y']=series
df4.to_csv('data\SEA_Gradual_high.csv')

dataset_5 = synth.ConceptDriftStream(stream=synth.SEA(seed=42, variant=3), drift_stream=synth.SEA(seed=42, variant=2), seed=1, position=250000, width=100000)
SEA_grad = dataset_5.take(500000)
frame = []
series = []
for x,y in SEA_grad:
    frame.append(x)
    series.append(y)

df5= pd.DataFrame(frame)
df5['y']=series
df5.to_csv('data\SEA_Gradual_100k_high.csv')

dataset_6 = synth.ConceptDriftStream(stream=synth.SEA(seed=42, variant=0), drift_stream=synth.SEA(seed=42, variant=1), seed=1, position=250000, width=1)
SEA_abrupt = dataset_6.take(500000)
frame = []
series = []
for x,y in SEA_abrupt:
    frame.append(x)
    series.append(y)

df6 = pd.DataFrame(frame)
df6['y']=series
df6.to_csv('data\SEA_Abrupt_low.csv')

dataset_7 = synth.ConceptDriftStream(stream=synth.SEA(seed=42, variant=2), drift_stream=synth.SEA(seed=42, variant=1), seed=1, position=250000, width=100000)
SEA_grad = dataset_7.take(500000)
frame = []
series = []
for x,y in SEA_grad:
    frame.append(x)
    series.append(y)

df7 = pd.DataFrame(frame)
df7['y']=series
df7.to_csv('data\SEA_Gradual_100k_low.csv')

dataset_8 = synth.Hyperplane(seed=42,n_drift_features=10,mag_change=0.8,noise_percentage=0,sigma=0.1)
# Gradual drift using the Hyperplane dataset. Currently set for high gradual drift. Adjust parameter settings as needed.
Hyperplane_grad_01 = dataset_8.take(500000)
frame=[]
series=[]
for x,y in Hyperplane_grad_01:
    frame.append(x)
    series.append(y)

df8 = pd.DataFrame(frame)
df8['y']=series
df8.to_csv('data\Hyperplane_01_high.csv')

dataset_9 = synth.Hyperplane(seed=42,n_drift_features=5,mag_change=0.1,noise_percentage=0,sigma=0.1)
# Gradual drift using the Hyperplane dataset. Currently set for high gradual drift. Adjust parameter settings as needed.
Hyperplane_grad_01 = dataset_9.take(500000)
frame=[]
series=[]
for x,y in Hyperplane_grad_01:
    frame.append(x)
    series.append(y)

df9 = pd.DataFrame(frame)
df9['y']=series
df9.to_csv('data\Hyperplane_01_5att.csv')


dataset_10 = synth.Hyperplane(seed=42,n_drift_features=10,mag_change=0.1,noise_percentage=0,sigma=0.001)
# Gradual drift using the Hyperplane dataset. Currently set for high gradual drift. Adjust parameter settings as needed.
Hyperplane_grad_01 = dataset_10.take(500000)
frame=[]
series=[]
for x,y in Hyperplane_grad_01:
    frame.append(x)
    series.append(y)

df10 = pd.DataFrame(frame)
df10['y']=series
df10.to_csv('data\Hyperplane_01_low.csv')
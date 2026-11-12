# Hinglish Viva Guide

## Quick explanation

"BusinessPulse Detective selected dataset ko reference baseline ke against compare karta hai. UI source ko user-uploaded, bundled sample, generated sample, ya synthetic demo ke roop mein label karta hai. User upload hone se business provenance verify nahi hoti; isliye us limitation ko display aur report dono mein likha jata hai."

## 3-line answer

1. "Main project ek A/B testing dashboard hai, but main feature ko anomaly detection workflow ke saath enhance kiya hai."
2. "Jab Control aur Treatment dono available hote hain, har Treatment segment ko Control ke against compare karte hain. Expected conversions = segment ke valid observations × Control conversion rate."
3. "30 observations ka default sirf configurable screening cutoff hai, adequate statistical power ka proof nahi. Repeated user/session IDs hon toh independent-observation inference ko suppress kiya jata hai."

## Viva points

- Baseline is Control conversion when both experiment groups are available; otherwise leave-one-segment-out reference is used.
- Bonferroni adjusted p-values multiple segment comparisons ko account karte hain, lekin significance causality prove nahi karti.
- Included segments ke excess conversions ko full aggregate change se alag report kiya jata hai. Missing ya low-volume segment ka residual silently redistribute nahi hota.
- Sample size threshold low-volume screening ke liye hai; actual power, repeated observations, aur cluster design ka alag analysis chahiye.
- The app still preserves the original A/B analytics dashboard, charts, filters, and report flow.
- Synthetic data is used only in labeled demos and tests; uploaded file ka business provenance independently verified nahi hota.
- The goal is to catch meaningful deviations, not to invent business conclusions.
- A/B test independent-row assumption ke under calculate hota hai. Agar user/session IDs repeat hon, independent-row test suppress hota hai. Unique ID ya ID ka absent hona independence ka proof nahi; clustered/paired design ke liye specialized method chahiye.
- Power planner specified baseline rate, target rate, alpha, desired power, aur sidedness se equal-allocation normal approximation use karta hai. Zero effect/boundary rates par result unavailable hota hai. Yeh planning estimate hai, actual experiment ki achieved power ka proof nahi.
- Uploaded file ka name, time, dimensions, missing counts, inferred types, size aur SHA-256 byte fingerprint report ho sakta hai. Hash sirf bytes identify karne mein help karta hai; business source/provenance verify nahi karta.
- Smart Alerts use only BusinessPulse Detective ke existing adjusted-significant results; koi second detection formula nahi hai. Severity observed absolute percentage-point difference ke configurable bands se aati hai. Yeh operational threshold hai, scientific ya universal severity proof nahi.
- Alert status existing project database mein save hota hai. KPI Drill-Down wahi baseline, segment calculations, contribution accounting, aur residual reuse karta hai.
- Bonferroni correction current segment dimension ke andar hota hai; multiple dimensions explore karne par un sab ka joint correction nahi hota.

## Short interviewer answer

"Pehle app data source aur provenance limitation batata hai. Phir Control reference se observed/expected conversions, lift aur adjusted p-value calculate karta hai. Repeated user/session IDs hon toh independent-row test aur Detective alert suppress hote hain; unique IDs independence prove nahi karte. Power planner sirf specified assumptions ke liye approximate future sample-size planning deta hai—yeh current study ki achieved power ka proof nahi. Segment contributions aggregate se reconcile na hon toh residual explicitly dikhaya jata hai. Adjusted significance causality prove nahi karti."

## Smart Alerts and Drill-Down

"Alert Center mein alerts existing Detective ke adjusted-significant rows se bante hain. Severity absolute percentage-point change ke user-configured bands ke mutabik hoti hai, aur screen rule bhi dikhati hai. Alert ka status existing database mein persist hota hai. KPI Drill-Down observed aur expected rates, segment counts, valid contribution aur aggregate residual dikhata hai. Yeh evidence exploratory hai—causality, power, ya observations ki independence prove nahi karta."

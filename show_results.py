import json

with open('privacy_shield_results.json') as f:
    data = json.load(f)

print('PRIVACY SHIELD - DEMONSTRATION RESULTS')
print('='*60)
print('\nBASELINE MODEL (No Privacy Protection):')
print(f"  F1 Score:           {data['baseline']['metrics']['f1']:.4f}")
print(f"  Accuracy:           {data['baseline']['metrics']['accuracy']:.4f}")
print(f"  ROC-AUC:            {data['baseline']['metrics']['roc_auc']:.4f}")
print(f"  MIA Attack Acc:     {data['baseline']['mia_attack']['attack_accuracy']:.4f}")

print('\nPROTECTED MODEL (With Privacy Shield):')
print(f"  F1 Score:           {data['protected']['metrics']['f1']:.4f}")
print(f"  Accuracy:           {data['protected']['metrics']['accuracy']:.4f}")
print(f"  ROC-AUC:            {data['protected']['metrics']['roc_auc']:.4f}")
print(f"  MIA Attack Acc:     {data['protected']['mia_attack']['attack_accuracy']:.4f}")

print('\nCOMPARISON:')
print(f"  Utility Change (F1):        {data['comparison']['utility_change_f1']:+.2f}%")
print(f"  Privacy Improvement (MIA): {data['comparison']['privacy_improvement_mia']:+.2f}%")

if 'epsilon' in data['comparison']:
    print('\nFORMAL PRIVACY GUARANTEE:')
    print(f"  Epsilon:            {data['comparison']['epsilon']:.4f}")
    print(f"  Delta:              {data['comparison']['delta']:.2e}")

print('\n' + '='*60)


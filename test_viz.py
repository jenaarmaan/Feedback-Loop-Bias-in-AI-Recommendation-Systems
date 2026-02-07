import numpy as np
from simulator import run_simulation, calculate_gini, calculate_diversity

def test():
    params = {
        "num_items": 50,
        "num_users": 20,
        "cycles": 10,
        "k": 3,
        "feedback_loop_strength": 0.5
    }
    
    print("Testing Naive Simulation...")
    history, items = run_simulation(**params, mitigation=False)
    
    last_snapshot = history[-1]
    exposures = [s[1] for s in last_snapshot['item_stats']]
    gini = calculate_gini(exposures)
    div = calculate_diversity(last_snapshot['item_stats'])
    
    print(f"Gini: {gini:.4f}, Diversity: {div}")
    assert gini >= 0 and gini <= 1
    assert div <= 50
    
    print("Testing Mitigated Simulation...")
    m_history, m_items = run_simulation(**params, mitigation=True)
    m_last_snapshot = m_history[-1]
    m_gini = calculate_gini([s[1] for s in m_last_snapshot['item_stats']])
    
    print(f"Mitigated Gini: {m_gini:.4f}")
    print("All tests passed!")

if __name__ == "__main__":
    test()

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy_financial import irr, pmt, npv
import io
import plotly.express as px

st.set_page_config(layout="wide")

# Initialize session state
if 'base_results' not in st.session_state:
    st.session_state.base_results = None
if 'last_capacity' not in st.session_state:
    st.session_state.last_capacity = None

# Define capex_distribution globally
capex_distribution = [0.4, 0.3, 0.3]

# Sidebar
st.sidebar.title("FLNG Project Configuration")
capacity_option = st.sidebar.selectbox("Select FLNG Capacity:", ["0.65 MTPA", "1.3 MTPA"])

# Clear session state if capacity changes
if st.session_state.last_capacity != capacity_option:
    st.session_state.base_results = None
    st.session_state.last_capacity = capacity_option

# Default values
if capacity_option == "0.65 MTPA":
    capex_default = 550.0
    daily_feed = 3000000.0
    staff_default = 200
    lng_mmbtu = 30558.904
    condensate_ton = 13000.0
    ngl_ton = 30000.0
else:
    capex_default = 1000.0
    daily_feed = 6000000.0
    staff_default = 220
    lng_mmbtu = 2 * 30558.904
    condensate_ton = 26000.0
    ngl_ton = 60000.0

# Create 3 columns
col1, col2, col3 = st.columns(3)

# --- CAPEX ---
with col1:
    st.header("CAPEX")
    capex = st.number_input("Total CAPEX (MUSD)", value=float(capex_default), min_value=0.0)
    build_duration = st.number_input("Construction Duration (Years)", value=3, min_value=1)
    operation_duration = st.number_input("Operation Duration (Years)", value=15, min_value=1)
    
    financing = st.checkbox("Financed?")
    if financing:
        interest_rate = st.number_input("Interest Rate (%)", value=14.0) / 100
        financed_percent = st.number_input("Financed Percent of CAPEX (%)", value=60.0) / 100
        repayment_period = st.number_input("Repayment Period (Years)", value=5, min_value=1)

# --- OPEX ---
with col2:
    st.header("OPEX")
    feed_price = st.number_input("Feed Price ($/m³)", value=0.02)
    feed_cost = feed_price * daily_feed * 365 / 1000000.0
    
    staff_count = st.number_input("Staff Count", value=staff_default)
    staff_salary = st.number_input("Monthly Salary ($)", value=3200.0)
    human_cost = staff_count * staff_salary * 12 / 1000000.0

    maintenance_pct = st.number_input("Maintenance (% CAPEX)", value=3.0)
    maintenance_cost = capex * maintenance_pct / 100.0

    insurance_pct = st.number_input("Insurance (% CAPEX)", value=1.5)
    insurance_cost = capex * insurance_pct / 100.0

    shipping_cost_unit = st.number_input("Shipping Cost ($/MMBTU)", value=1.2)
    shipping_cost = shipping_cost_unit * lng_mmbtu / 1000.0

    opex_subtotal = feed_cost + human_cost + maintenance_cost + insurance_cost + shipping_cost
    other_pct = st.number_input("Other OPEX (% of subtotal)", value=22.5)
    other_cost = opex_subtotal * other_pct / 100.0
    total_opex = opex_subtotal + other_cost

# --- Revenue ---
with col3:
    st.header("Revenue")
    lng_price = st.number_input("LNG Price ($/MMBTU)", value=10.0)
    cond_price = st.number_input("Condensate Price ($/ton)", value=650.0)
    ngl_price = st.number_input("NGL Price ($/ton)", value=400.0)

    lng_income = lng_price * lng_mmbtu / 1000.0
    cond_income = cond_price * condensate_ton / 1000000.0
    ngl_income = ngl_price * ngl_ton / 1000000.0
    total_revenue = lng_income + cond_income + ngl_income

# --- Sensitivity Analysis ---
st.sidebar.subheader("Sensitivity Analysis")
multi_var = st.sidebar.checkbox("Multi-Variable Sensitivity")
discount_rate = st.sidebar.number_input("Discount Rate for NPV (%)", value=10.0, min_value=0.0) / 100

variables = ["CAPEX", "OPEX", "Operation Duration"] + (["Interest Rate", "Financed Percent", "Repayment Period"] if financing else [])
if multi_var:
    var1 = st.sidebar.selectbox("First Variable", variables)
    var2 = st.sidebar.selectbox("Second Variable", [v for v in variables if v != var1])
else:
    sensitivity_var = st.sidebar.selectbox("Select Variable for Sensitivity Analysis:", variables)

# Sensitivity range inputs
if not multi_var:
    if sensitivity_var == "CAPEX":
        capex_min = st.sidebar.number_input("CAPEX Min (MUSD)", value=float(max(0, capex - 100)), min_value=0.0)
        capex_max = st.sidebar.number_input("CAPEX Max (MUSD)", value=float(capex + 100))
        capex_step = st.sidebar.number_input("CAPEX Step (MUSD)", value=20.0, min_value=1.0)
    elif sensitivity_var == "OPEX":
        opex_min = st.sidebar.number_input("OPEX Min (MUSD)", value=float(max(0, total_opex - 50)), min_value=0.0)
        opex_max = st.sidebar.number_input("OPEX Max (MUSD)", value=float(total_opex + 50))
        opex_step = st.sidebar.number_input("OPEX Step (MUSD)", value=10.0, min_value=1.0)
    elif sensitivity_var == "Operation Duration":
        op_duration_min = st.sidebar.number_input("Operation Duration Min (Years)", value=10, min_value=1)
        op_duration_max = st.sidebar.number_input("Operation Duration Max (Years)", value=20)
        op_duration_step = st.sidebar.number_input("Operation Duration Step (Years)", value=2, min_value=1)
else:
    if var1 == "CAPEX":
        capex1_min = st.sidebar.number_input("CAPEX Min (MUSD)", value=float(max(0, capex - 100)), min_value=0.0)
        capex1_max = st.sidebar.number_input("CAPEX Max (MUSD)", value=float(capex + 100))
        capex1_step = st.sidebar.number_input("CAPEX Step (MUSD)", value=20.0, min_value=1.0)
    elif var1 == "OPEX":
        opex1_min = st.sidebar.number_input("OPEX Min (MUSD)", value=float(max(0, total_opex - 50)), min_value=0.0)
        opex1_max = st.sidebar.number_input("OPEX Max (MUSD)", value=float(total_opex + 50))
        opex1_step = st.sidebar.number_input("OPEX Step (MUSD)", value=10.0, min_value=1.0)
    elif var1 == "Operation Duration":
        op_duration1_min = st.sidebar.number_input("Operation Duration Min (Years)", value=10, min_value=1)
        op_duration1_max = st.sidebar.number_input("Operation Duration Max (Years)", value=20)
        op_duration1_step = st.sidebar.number_input("Operation Duration Step (Years)", value=2, min_value=1)
    elif var1 == "Interest Rate" and financing:
        int_rate1_min = st.sidebar.number_input("Interest Rate Min (%)", value=5.0, min_value=0.0)
        int_rate1_max = st.sidebar.number_input("Interest Rate Max (%)", value=20.0)
        int_rate1_step = st.sidebar.number_input("Interest Rate Step (%)", value=3.0, min_value=0.1)
    elif var1 == "Financed Percent" and financing:
        fin_pct1_min = st.sidebar.number_input("Financed Percent Min (%)", value=0.0, min_value=0.0)
        fin_pct1_max = st.sidebar.number_input("Financed Percent Max (%)", value=80.0)
        fin_pct1_step = st.sidebar.number_input("Financed Percent Step (%)", value=20.0, min_value=1.0)
    elif var1 == "Repayment Period" and financing:
        rep_period1_min = st.sidebar.number_input("Repayment Period Min (Years)", value=2, min_value=1)
        rep_period1_max = st.sidebar.number_input("Repayment Period Max (Years)", value=8)
        rep_period1_step = st.sidebar.number_input("Repayment Period Step (Years)", value=2, min_value=1)

    if var2 == "CAPEX":
        capex2_min = st.sidebar.number_input("CAPEX Min (MUSD)", value=float(max(0, capex - 100)), min_value=0.0)
        capex2_max = st.sidebar.number_input("CAPEX Max (MUSD)", value=float(capex + 100))
        capex2_step = st.sidebar.number_input("CAPEX Step (MUSD)", value=20.0, min_value=1.0)
    elif var2 == "OPEX":
        opex2_min = st.sidebar.number_input("OPEX Min (MUSD)", value=float(max(0, total_opex - 50)), min_value=0.0)
        opex2_max = st.sidebar.number_input("OPEX Max (MUSD)", value=float(total_opex + 50))
        opex2_step = st.sidebar.number_input("OPEX Step (MUSD)", value=10.0, min_value=1.0)
    elif var2 == "Operation Duration":
        op_duration2_min = st.sidebar.number_input("Operation Duration Min (Years)", value=10, min_value=1)
        op_duration2_max = st.sidebar.number_input("Operation Duration Max (Years)", value=20)
        op_duration2_step = st.sidebar.number_input("Operation Duration Step (Years)", value=2, min_value=1)
    elif var2 == "Interest Rate" and financing:
        int_rate2_min = st.sidebar.number_input("Interest Rate Min (%)", value=5.0, min_value=0.0)
        int_rate2_max = st.sidebar.number_input("Interest Rate Max (%)", value=20.0)
        int_rate2_step = st.sidebar.number_input("Interest Rate Step (%)", value=3.0, min_value=0.1)
    elif var2 == "Financed Percent" and financing:
        fin_pct2_min = st.sidebar.number_input("Financed Percent Min (%)", value=0.0, min_value=0.0)
        fin_pct2_max = st.sidebar.number_input("Financed Percent Max (%)", value=80.0)
        fin_pct2_step = st.sidebar.number_input("Financed Percent Step (%)", value=20.0, min_value=1.0)
    elif var2 == "Repayment Period" and financing:
        rep_period2_min = st.sidebar.number_input("Repayment Period Min (Years)", value=2, min_value=1)
        rep_period2_max = st.sidebar.number_input("Repayment Period Max (Years)", value=8)
        rep_period2_step = st.sidebar.number_input("Repayment Period Step (Years)", value=2, min_value=1)

# --- Results ---
st.markdown("---")
st.header("📊 Financial Results")

# Base case calculation
if st.button("📈 Calculate"):
    cash_flows_base = []
    if build_duration > len(capex_distribution):
        remaining_capex = 1.0 - sum(capex_distribution)
        capex_distribution.extend([remaining_capex/(build_duration - len(capex_distribution))] * 
                                (build_duration - len(capex_distribution)))
    
    for i in range(int(build_duration)):
        portion = capex * capex_distribution[i] if i < len(capex_distribution) else 0.0
        cash_flows_base.append(-abs(portion))

    loan_payment = 0
    total_interest_paid = 0
    if financing:
        financed_amount = capex * financed_percent
        loan_payment = -pmt(interest_rate, repayment_period, financed_amount)
        total_interest_paid = loan_payment * repayment_period - financed_amount

    annual_profit_before_loan = total_revenue - total_opex
    
    if annual_profit_before_loan <= 0:
        st.error("❌ Operation years show no profit (Revenue <= OPEX). Cannot calculate IRR.")
        st.session_state.base_results = None
    else:
        for year in range(int(operation_duration)):
            if financing and year < repayment_period:
                net_cash_flow = annual_profit_before_loan - loan_payment
                cash_flows_base.append(net_cash_flow)
            else:
                cash_flows_base.append(annual_profit_before_loan)

        if any(cf < 0 for cf in cash_flows_base) and any(cf > 0 for cf in cash_flows_base):
            try:
                irr_value = irr(cash_flows_base)
                npv_value = npv(discount_rate, cash_flows_base)
                if not np.isnan(irr_value):
                    st.success(f"**IRR:** {irr_value * 100:.2f}%")
                    st.success(f"**NPV (at {discount_rate*100:.1f}%):** {npv_value:.2f} MUSD")
                    adjusted_capex = capex + (total_interest_paid if financing else 0)
                    avg_annual_profit = total_revenue - total_opex - (adjusted_capex / operation_duration)
                    st.success(f"**Average Annual Profit:** {avg_annual_profit:.2f} MUSD")
                    st.info(f"**Total OPEX:** {total_opex:.2f} MUSD")
                    st.info(f"**Total Revenue:** {total_revenue:.2f} MUSD")
                    if financing:
                        st.info(f"**Total Interest Paid:** {total_interest_paid:.2f} MUSD")
                    
                    cumulative_cash = np.cumsum(cash_flows_base)
                    break_even_idx = next((i for i, x in enumerate(cumulative_cash) if x >= 0), None)
                    if break_even_idx is not None:
                        st.info(f"**Break-Even Year:** {break_even_idx + 1 - 3}")
                        if break_even_idx > 0:
                            prev_cumulative = cumulative_cash[break_even_idx - 1]
                            curr_cumulative = cumulative_cash[break_even_idx]
                            break_even_fraction = -prev_cumulative / (curr_cumulative - prev_cumulative) if curr_cumulative != prev_cumulative else 0
                        else:
                            break_even_fraction = 0

                    # --- Charts Visualization (Side by Side) ---
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 2]})

                    # --- OPEX Pie Chart (Left) ---
                    opex_components = {
                        "Feed Cost": feed_cost,
                        "Human Cost": human_cost,
                        "Maintenance Cost": maintenance_cost,
                        "Insurance Cost": insurance_cost,
                        "Shipping Cost": shipping_cost,
                        "Other Cost": other_cost
                    }
                    labels = [key for key, value in opex_components.items() if value > 0]
                    sizes = [value for value in opex_components.values() if value > 0]
                    
                    if sizes:
                        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired(np.arange(len(labels))))
                        ax1.axis('equal')
                        ax1.set_title("OPEX Cost Distribution (MUSD)")
                    else:
                        ax1.text(0.5, 0.5, "No OPEX costs to display", ha='center', va='center')

                    # --- Cash Flow Visualization (Right) ---
                    years = range(1, len(cash_flows_base)+1)
                    capex_values = []
                    opex_values = []
                    loan_values = []
                    revenue_values = []
                    
                    for i, cf in enumerate(cash_flows_base):
                        if i < build_duration:
                            capex_values.append(cf)
                            opex_values.append(0)
                            loan_values.append(0)
                            revenue_values.append(0)
                        else:
                            capex_values.append(0)
                            opex_values.append(-total_opex)
                            if financing and (i - build_duration) < repayment_period:
                                loan_values.append(-loan_payment)
                            else:
                                loan_values.append(0)
                            revenue_values.append(total_revenue)
                    
                    bottom = np.zeros(len(years))
                    for i, year in enumerate(years):
                        if capex_values[i] < 0:
                            ax2.bar(year, capex_values[i], color='red', bottom=bottom[i])
                            bottom[i] += capex_values[i]
                        if opex_values[i] < 0:
                            ax2.bar(year, opex_values[i], color='orange', bottom=bottom[i])
                            bottom[i] += opex_values[i]
                        if loan_values[i] < 0:
                            ax2.bar(year, loan_values[i], color='darkred', bottom=bottom[i])
                            bottom[i] += loan_values[i]
                    
                    for i, year in enumerate(years):
                        if revenue_values[i] > 0:
                            if break_even_idx is not None and i == break_even_idx:
                                pre_break_even_height = revenue_values[i] * break_even_fraction
                                post_break_even_height = revenue_values[i] * (1 - break_even_fraction)
                                ax2.bar(year, pre_break_even_height, color='lightgreen', bottom=0, label='Revenue (Pre-BreakEven)' if i == build_duration else "")
                                ax2.bar(year, post_break_even_height, color='darkgreen', bottom=pre_break_even_height, label='Revenue (Post-BreakEven)' if i == build_duration else "")
                            elif break_even_idx is not None and i > break_even_idx:
                                ax2.bar(year, revenue_values[i], color='darkgreen', bottom=0, label='Revenue (Post-BreakEven)' if i == build_duration else "")
                            else:
                                ax2.bar(year, revenue_values[i], color='lightgreen', bottom=0, label='Revenue (Pre-BreakEven)' if i == build_duration else "")
                    
                    if break_even_idx is not None:
                        ax2.axvline(x=break_even_idx+1, color='blue', linestyle='--', linewidth=1)
                        ax2.text(break_even_idx+1, ax2.get_ylim()[1]*0.9, 
                                f'Break-Even Year {break_even_idx+1}', 
                                ha='center', color='blue')
                    
                    ax2.axhline(0, color='black', linewidth=0.5)
                    ax2.set_xlabel("Year")
                    ax2.set_ylabel("Cash Flow (MUSD)")
                    ax2.set_title("Project Cash Flow (Negative and Positive Values)")
                    
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='red', label='CAPEX'),
                        Patch(facecolor='orange', label='OPEX'),
                        Patch(facecolor='darkred', label='Loan Payments'),
                        Patch(facecolor='lightgreen', label='Revenue (Pre-BreakEven)'),
                        Patch(facecolor='darkgreen', label='Revenue (Post-BreakEven)')
                    ]
                    ax2.legend(handles=legend_elements, loc='upper left')
                    st.pyplot(fig)

                    # --- Cash Flow Table ---
                    st.subheader("Detailed Cash Flows")
                    year_types = []
                    for i in range(len(cash_flows_base)):
                        if i < build_duration:
                            year_types.append("Construction")
                        elif financing and (i - build_duration) < repayment_period:
                            year_types.append("Operation (with Loan)")
                        else:
                            year_types.append("Operation")
                    
                    st.table({
                        "Year": list(years),
                        "Type": year_types,
                        "Net Cash Flow (MUSD)": [f"{cf:.2f}" for cf in cash_flows_base],
                        "Cumulative Cash Flow": [f"{cum:.2f}" for cum in cumulative_cash]
                    })

                    # Store results in session state
                    st.session_state.base_results = {
                        "cash_flows_base": cash_flows_base,
                        "irr_value": irr_value,
                        "npv_value": npv_value,
                        "avg_annual_profit": avg_annual_profit,
                        "total_opex": total_opex,
                        "total_revenue": total_revenue,
                        "total_interest_paid": total_interest_paid if financing else 0,
                        "cumulative_cash": cumulative_cash,
                        "break_even_idx": break_even_idx,
                        "break_even_fraction": break_even_fraction,
                        "fig": fig,
                        "years": years,
                        "year_types": year_types,
                        "capacity": capacity_option
                    }
            except Exception as e:
                st.error(f"❌ Calculation error: {str(e)}")
                st.session_state.base_results = None
        else:
            st.error("❌ IRR requires both negative (CAPEX) and positive (Operation) cash flows")
            st.session_state.base_results = None

# Display base results if they exist and match the current capacity
if st.session_state.base_results and st.session_state.base_results.get("capacity") == capacity_option:
    st.success(f"**IRR:** {st.session_state.base_results['irr_value'] * 100:.2f}%")
    st.success(f"**NPV (at {discount_rate*100:.1f}%):** {st.session_state.base_results['npv_value']:.2f} MUSD")
    st.success(f"**Average Annual Profit:** {st.session_state.base_results['avg_annual_profit']:.2f} MUSD")
    st.info(f"**Total OPEX:** {st.session_state.base_results['total_opex']:.2f} MUSD")
    st.info(f"**Total Revenue:** {st.session_state.base_results['total_revenue']:.2f} MUSD")
    if financing:
        st.info(f"**Total Interest Paid:** {st.session_state.base_results['total_interest_paid']:.2f} MUSD")
    
    if st.session_state.base_results['break_even_idx'] is not None:
        st.info(f"**Break-Even Year:** {st.session_state.base_results['break_even_idx'] + 1 - 3}")
    
    st.pyplot(st.session_state.base_results['fig'])
    
    st.subheader("Detailed Cash Flows")
    st.table({
        "Year": st.session_state.base_results['years'],
        "Type": st.session_state.base_results['year_types'],
        "Net Cash Flow (MUSD)": [f"{cf:.2f}" for cf in st.session_state.base_results['cash_flows_base']],
        "Cumulative Cash Flow": [f"{cum:.2f}" for cum in st.session_state.base_results['cumulative_cash']]
    })

# Sensitivity Analysis
st.subheader("Sensitivity Analysis")
run_sensitivity = st.checkbox("Run Sensitivity Analysis")
if run_sensitivity:
    if not st.session_state.base_results or st.session_state.base_results.get("capacity") != capacity_option:
        st.warning("Please calculate the base case for the current capacity by clicking 'Calculate'.")
    elif not multi_var and not sensitivity_var:
        st.warning("Please select a variable for sensitivity analysis.")
    elif multi_var and (not var1 or not var2):
        st.warning("Please select two variables for multi-variable sensitivity analysis.")
    else:
        try:
            cash_flows_base = st.session_state.base_results['cash_flows_base']
            annual_profit_before_loan = st.session_state.base_results['total_revenue'] - st.session_state.base_results['total_opex']
            loan_payment = -pmt(interest_rate, repayment_period, capex * financed_percent) if financing else 0
            total_interest_paid = loan_payment * repayment_period - (capex * financed_percent) if financing else 0

            if not multi_var:
                sensitivity_results = []
                
                if sensitivity_var == "CAPEX":
                    values = np.arange(capex_min, capex_max + capex_step, capex_step)
                    for val in values:
                        temp_cash_flows = []
                        if build_duration > len(capex_distribution):
                            remaining_capex = 1.0 - sum(capex_distribution)
                            temp_distribution = capex_distribution + [remaining_capex/(build_duration - len(capex_distribution))] * (build_duration - len(capex_distribution))
                        else:
                            temp_distribution = capex_distribution
                        for i in range(int(build_duration)):
                            portion = val * temp_distribution[i] if i < len(temp_distribution) else 0.0
                            temp_cash_flows.append(-abs(portion))
                        temp_maintenance = val * maintenance_pct / 100.0
                        temp_insurance = val * insurance_pct / 100.0
                        temp_opex = (feed_cost + human_cost + temp_maintenance + temp_insurance + shipping_cost) * (1 + other_pct/100)
                        temp_loan_pmt = -pmt(interest_rate, repayment_period, val * financed_percent) if financing else 0
                        temp_profit = total_revenue - temp_opex
                        for year in range(int(operation_duration)):
                            if financing and year < repayment_period:
                                temp_cash_flows.append(temp_profit - temp_loan_pmt)
                            else:
                                temp_cash_flows.append(temp_profit)
                        temp_irr = irr(temp_cash_flows)*100 if any(cf < 0 for cf in temp_cash_flows) and any(cf > 0 for cf in temp_cash_flows) else np.nan
                        temp_npv = npv(discount_rate, temp_cash_flows)
                        temp_total_int = temp_loan_pmt * repayment_period - (val * financed_percent) if financing else 0
                        temp_avg_profit = total_revenue - temp_opex - ((val + temp_total_int) / operation_duration)
                        sensitivity_results.append((val, temp_irr, temp_npv, temp_avg_profit))
                
                elif sensitivity_var == "OPEX":
                    values = np.arange(opex_min, opex_max + opex_step, opex_step)
                    for val in values:
                        temp_cash_flows = cash_flows_base[:build_duration]
                        temp_profit = total_revenue - val
                        for year in range(int(operation_duration)):
                            if financing and year < repayment_period:
                                temp_cash_flows.append(temp_profit - loan_payment)
                            else:
                                temp_cash_flows.append(temp_profit)
                        temp_irr = irr(temp_cash_flows)*100 if any(cf < 0 for cf in temp_cash_flows) and any(cf > 0 for cf in temp_cash_flows) else np.nan
                        temp_npv = npv(discount_rate, temp_cash_flows)
                        temp_avg_profit = total_revenue - val - ((capex + total_interest_paid) / operation_duration)
                        sensitivity_results.append((val, temp_irr, temp_npv, temp_avg_profit))
                
                elif sensitivity_var == "Operation Duration":
                    values = np.arange(op_duration_min, op_duration_max + op_duration_step, op_duration_step)
                    for val in values:
                        temp_cash_flows = cash_flows_base[:build_duration]
                        for year in range(int(val)):
                            if financing and year < repayment_period:
                                temp_cash_flows.append(annual_profit_before_loan - loan_payment)
                            else:
                                temp_cash_flows.append(annual_profit_before_loan)
                        temp_irr = irr(temp_cash_flows)*100 if any(cf < 0 for cf in temp_cash_flows) and any(cf > 0 for cf in temp_cash_flows) else np.nan
                        temp_npv = npv(discount_rate, temp_cash_flows)
                        temp_avg_profit = total_revenue - total_opex - ((capex + total_interest_paid) / val)
                        sensitivity_results.append((val, temp_irr, temp_npv, temp_avg_profit))
                
                elif sensitivity_var == "Interest Rate" and financing:
                    values = np.arange(0.05, 0.21, 0.03)
                    for val in values:
                        temp_loan_pmt = -pmt(val, repayment_period, capex * financed_percent)
                        temp_total_int = temp_loan_pmt * repayment_period - (capex * financed_percent)
                        temp_cash_flows = cash_flows_base[:build_duration]
                        for year in range(int(operation_duration)):
                            if year < repayment_period:
                                temp_cash_flows.append(annual_profit_before_loan - temp_loan_pmt)
                            else:
                                temp_cash_flows.append(annual_profit_before_loan)
                        temp_irr = irr(temp_cash_flows)*100 if any(cf < 0 for cf in temp_cash_flows) and any(cf > 0 for cf in temp_cash_flows) else np.nan
                        temp_npv = npv(discount_rate, temp_cash_flows)
                        temp_avg_profit = total_revenue - total_opex - ((capex + temp_total_int) / operation_duration)
                        sensitivity_results.append((val * 100, temp_irr, temp_npv, temp_avg_profit))
                
                elif sensitivity_var == "Financed Percent" and financing:
                    values = np.arange(0, 0.81, 0.20)
                    for val in values:
                        temp_financed = capex * val
                        temp_loan_pmt = -pmt(interest_rate, repayment_period, temp_financed)
                        temp_total_int = temp_loan_pmt * repayment_period - temp_financed
                        temp_cash_flows = cash_flows_base[:build_duration]
                        for year in range(int(operation_duration)):
                            if year < repayment_period:
                                temp_cash_flows.append(annual_profit_before_loan - temp_loan_pmt)
                            else:
                                temp_cash_flows.append(annual_profit_before_loan)
                        temp_irr = irr(temp_cash_flows)*100 if any(cf < 0 for cf in temp_cash_flows) and any(cf > 0 for cf in temp_cash_flows) else np.nan
                        temp_npv = npv(discount_rate, temp_cash_flows)
                        temp_avg_profit = total_revenue - total_opex - ((capex + temp_total_int) / operation_duration)
                        sensitivity_results.append((val * 100, temp_irr, temp_npv, temp_avg_profit))
                
                elif sensitivity_var == "Repayment Period" and financing:
                    values = np.arange(2, 9, 2)
                    for val in values:
                        temp_loan_pmt = -pmt(interest_rate, val, capex * financed_percent)
                        temp_total_int = temp_loan_pmt * val - (capex * financed_percent)
                        temp_cash_flows = cash_flows_base[:build_duration]
                        for year in range(int(operation_duration)):
                            if year < val:
                                temp_cash_flows.append(annual_profit_before_loan - temp_loan_pmt)
                            else:
                                temp_cash_flows.append(annual_profit_before_loan)
                        temp_irr = irr(temp_cash_flows)*100 if any(cf < 0 for cf in temp_cash_flows) and any(cf > 0 for cf in temp_cash_flows) else np.nan
                        temp_npv = npv(discount_rate, temp_cash_flows)
                        temp_avg_profit = total_revenue - total_opex - ((capex + temp_total_int) / operation_duration)
                        sensitivity_results.append((val, temp_irr, temp_npv, temp_avg_profit))
                
                if sensitivity_results:
                    df = pd.DataFrame(sensitivity_results, columns=[sensitivity_var, "IRR (%)", f"NPV (at {discount_rate*100:.1f}%)", "Average Annual Profit (MUSD)"])
                    st.write(f"Sensitivity to {sensitivity_var}:")
                    styled_df = df.style.format({
                        sensitivity_var: "{:.2f}",
                        "IRR (%)": "{:.2f}%",
                        f"NPV (at {discount_rate*100:.1f}%)": "{:.2f}",
                        "Average Annual Profit (MUSD)": "{:.2f}"
                    })
                    st.dataframe(styled_df)

                    # Download button for CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Sensitivity Table as CSV",
                        data=csv,
                        file_name=f"sensitivity_{sensitivity_var.lower()}.csv",
                        mime="text/csv",
                    )

                    # Chart with enhancements
                    fig_sens, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(df[sensitivity_var], df["IRR (%)"], 'b-o', label="IRR (%)")
                    ax.set_xlabel(sensitivity_var)
                    ax.set_ylabel("IRR (%)", color='b')
                    ax.tick_params(axis='y', labelcolor='b')
                    ax.grid(True)
                    
                    # Add base case marker
                    base_value = {
                        "CAPEX": capex,
                        "OPEX": total_opex,
                        "Operation Duration": operation_duration,
                        "Interest Rate": interest_rate * 100,
                        "Financed Percent": financed_percent * 100,
                        "Repayment Period": repayment_period
                    }.get(sensitivity_var)
                    ax.axvline(x=base_value, color='g', linestyle='--', label='Base Case')

                    ax.legend()
                    st.pyplot(fig_sens)
                    plt.close(fig_sens)
                else:
                    st.warning("No sensitivity results generated for selected variable.")
            
            # Multi-variable sensitivity analysis
            else:
                # Generate value ranges
                if var1 == "CAPEX":
                    v1_values = np.arange(capex1_min, capex1_max + capex1_step, capex1_step)
                elif var1 == "OPEX":
                    v1_values = np.arange(opex1_min, opex1_max + opex1_step, opex1_step)
                elif var1 == "Operation Duration":
                    v1_values = np.arange(op_duration1_min, op_duration1_max + op_duration1_step, op_duration1_step)
                elif var1 == "Interest Rate" and financing:
                    v1_values = np.arange(int_rate1_min/100, int_rate1_max/100 + int_rate1_step/100, int_rate1_step/100)
                elif var1 == "Financed Percent" and financing:
                    v1_values = np.arange(fin_pct1_min/100, fin_pct1_max/100 + fin_pct1_step/100, fin_pct1_step/100)
                elif var1 == "Repayment Period" and financing:
                    v1_values = np.arange(rep_period1_min, rep_period1_max + rep_period1_step, rep_period1_step)

                if var2 == "CAPEX":
                    v2_values = np.arange(capex2_min, capex2_max + capex2_step, capex2_step)
                elif var2 == "OPEX":
                    v2_values = np.arange(opex2_min, opex2_max + opex2_step, opex2_step)
                elif var2 == "Operation Duration":
                    v2_values = np.arange(op_duration2_min, op_duration2_max + op_duration2_step, op_duration2_step)
                elif var2 == "Interest Rate" and financing:
                    v2_values = np.arange(int_rate2_min/100, int_rate2_max/100 + int_rate2_step/100, int_rate2_step/100)
                elif var2 == "Financed Percent" and financing:
                    v2_values = np.arange(fin_pct2_min/100, fin_pct2_max/100 + fin_pct2_step/100, fin_pct2_step/100)
                elif var2 == "Repayment Period" and financing:
                    v2_values = np.arange(rep_period2_min, rep_period2_max + rep_period2_step, rep_period2_step)

                # Create meshgrid
                v1_grid, v2_grid = np.meshgrid(v1_values, v2_values)
                irr_grid = np.zeros_like(v1_grid, dtype=float)

                # Calculate IRR for each combination
                for i in range(len(v1_values)):
                    for j in range(len(v2_values)):
                        temp_cash_flows = cash_flows_base[:int(build_duration)]
                        if var1 == "CAPEX":
                            if build_duration > len(capex_distribution):
                                remaining_capex = 1.0 - sum(capex_distribution)
                                temp_distribution = capex_distribution + [remaining_capex/(build_duration - len(capex_distribution))] * (build_duration - len(capex_distribution))
                            else:
                                temp_distribution = capex_distribution
                            for k in range(int(build_duration)):
                                portion = v1_grid[j, i] * temp_distribution[k] if k < len(temp_distribution) else 0.0
                                temp_cash_flows.append(-abs(portion))
                        if var2 == "CAPEX":
                            if build_duration > len(capex_distribution):
                                remaining_capex = 1.0 - sum(capex_distribution)
                                temp_distribution = capex_distribution + [remaining_capex/(build_duration - len(capex_distribution))] * (build_duration - len(capex_distribution))
                            else:
                                temp_distribution = capex_distribution
                            for k in range(int(build_duration)):
                                portion = v2_grid[j, i] * temp_distribution[k] if k < len(temp_distribution) else 0.0
                                temp_cash_flows.append(-abs(portion))
                        
                        temp_maintenance = (v1_grid[j, i] if var1 == "CAPEX" else capex) * maintenance_pct / 100.0
                        temp_insurance = (v1_grid[j, i] if var1 == "CAPEX" else capex) * insurance_pct / 100.0
                        temp_opex = (v2_grid[j, i] if var2 == "OPEX" else total_opex) if var2 == "OPEX" else (
                            (feed_cost + human_cost + temp_maintenance + temp_insurance + shipping_cost) * (1 + other_pct/100))
                        temp_loan_pmt = -pmt((v1_grid[j, i]/100 if var1 == "Interest Rate" else interest_rate),
                                            (v2_grid[j, i] if var2 == "Repayment Period" else repayment_period),
                                            (v1_grid[j, i]/100 if var1 == "Financed Percent" else financed_percent) * capex) if financing else 0
                        temp_profit = total_revenue - temp_opex
                        op_duration = v2_grid[j, i] if var2 == "Operation Duration" else operation_duration
                        for year in range(int(op_duration)):
                            if financing and year < (v2_grid[j, i] if var2 == "Repayment Period" else repayment_period):
                                temp_cash_flows.append(temp_profit - temp_loan_pmt)
                            else:
                                temp_cash_flows.append(temp_profit)
                        irr_grid[j, i] = irr(temp_cash_flows)*100 if any(cf < 0 for cf in temp_cash_flows) and any(cf > 0 for cf in temp_cash_flows) else np.nan

                # Create a DataFrame for Plotly
                df_heatmap = pd.DataFrame({
                    var1: v1_grid.flatten(),
                    var2: v2_grid.flatten(),
                    "IRR (%)": irr_grid.flatten()
                })

                # Display heatmap with Plotly
                fig_sens = px.imshow(
                    irr_grid,
                    x=[f"{x:.1f}" for x in v1_values],
                    y=[f"{y:.1f}" for y in v2_values],
                    labels={"x": var1, "y": var2, "color": "IRR (%)"},
                    title=f"IRR Heatmap: {var1} vs {var2}",
                    color_continuous_scale="Viridis",
                    range_color=[0, 20]
                )
                fig_sens.update_layout(
                    xaxis_title=var1,
                    yaxis_title=var2,
                    coloraxis_colorbar_title="IRR (%)"
                )
                st.plotly_chart(fig_sens)

        except Exception as e:
            st.error(f"Sensitivity analysis error: {str(e)}")

# import juliacall
# from juliacall import Main as jl
# from juliacall import Pkg as jlPkg
# import streamlit as st
# from streamlit_julia_call import julia_eval, julia_display

# if __name__ == "__main__":
    # with st.spinner("Initializing Julia runtime environment..."):
    #     julia_eval("using simulation")
    # jl = juliacall.newmodule("using MixedModels")
    # # jl = juliacall.newmodule("simulation")
    # jl.seval("fm = @formula(value ~ phase + (phase|subj_id));fm1 = fit(MixedModel, fm, df, REML=false);return coeftable(fm1).cols[4][2]")

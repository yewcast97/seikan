"""Compile a Thesis DSL into a forward-return event study.

Entry points live on the submodules: :func:`seikan.compiler.data.resolve_data_files` binds the
DSL's logical data keys to files and :func:`seikan.compiler.data.load_market_data` materializes
the aligned market frames once per run, and :func:`seikan.compiler.runner.run_backtest` measures
every declared parameter × horizon cell over the full sample against that one ``MarketData``.
"""

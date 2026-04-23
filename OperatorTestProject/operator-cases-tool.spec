# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['entry_point.py'],  # 使用独立的入口文件，而不是 src/main.py
    pathex=[],
    binaries=[],
    datas=[
        # 不包含任何数据文件，所有资源文件都在exe外部
    ],
    hiddenimports=[
        # src包及其所有子模块
        'src',
        'src.main',
        'src.config_loader',
        'src.llm_service',
        'src.module_processor',
        'src.skill_executor',
        'src.graph',
        'src.models',
        'src.exceptions',
        'src.path_utils',
        'src.test_case_generator',
        'src.common_model_definition',
        'src.error_merger',
        'src.extraction_node',
        'src.json_to_model_loader',
        'src.json_validation_node',
        'src.module_graph_state',
        'src.module_processing_graph',
        'src.prompt_builder',
        'src.result_saver',
        'src.rule_loader',
        'src.validation_node',
        # PyInstaller 可能无法自动检测的依赖
        'langgraph',
        'langchain_anthropic',
        'pydantic',
        'yaml',
        'aiofiles',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='operator-cases-tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # 保持控制台窗口，方便查看日志
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
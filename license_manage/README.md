位置: F:\RSNA\medical_imaging_workflow\license_manage

运行:
1) python -m license_manage.main
2) 输入密钥、确认CPU指纹与授权截止时间，点击生成

说明:
- CPU指纹自动读取，优先 Win32_Processor.ProcessorId
- License为本地签发的签名令牌，包含CPU与到期时间
- 授权记录存储在 licenses.db

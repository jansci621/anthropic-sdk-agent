# 天气 API 参考

## wttr.in API

免费天气 API，无需 API key。

- 查询格式: `https://wttr.in/{city}?format=j1`
- 返回 JSON 格式天气数据
- 支持中文城市名

### 示例请求

```
https://wttr.in/北京?format=j1
```

### 关键字段

- `current_condition[0].temp_C` — 当前温度
- `current_condition[0].weatherDesc[0].value` — 天气描述
- `current_condition[0].windspeedKmph` — 风速
- `current_condition[0].humidity` — 湿度
- `current_condition[0].FeelsLikeC` — 体感温度

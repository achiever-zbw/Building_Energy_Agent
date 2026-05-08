# Tool Usage Notes

函数名与参数由系统自动提供给模型；本节说明 **易混点与推荐用法**

## 通用

- **`building_id`**：必须与数据库中的建筑主键一致（区分大小写）；用户口述名称可能不等于 ID，必要时先 `get_building_basic_info` 核对。
- **`start_time` / `end_time`**：字符串形式，由服务端 `transform_timestamp` 解析；可与 MCP 时代相同的写法保持一致（例如 `2016-01-01 00:00:00`）。
- **能耗字段**：返回 JSON 中的英文键：`electricity`、`chilledwater`、`hotwater`、`water`；对用户说明时可写「电力 / 冷冻水 / …」。

## `get_building_basic_info`

- 用途：建筑面积、用途类型、建成年份、是否具备各类表计等。
- 适用于：首次接触某建筑 ID，或判断能否查电 / 水等。

## `get_building_time_energy`

- 用途：一段时间内的 **各类型累计能耗**（不是逐时明细）。
- 典型：**某年全年** → `start_time` / `end_time` 取自然年起止。

## `get_building_time_energy_by_hour`

- 用途：按 **小时桶** 聚合，便于曲线与峰谷分析；返回 `series` 列表。
- 区间过长时数据量会变大，可向用户说明「仅展示部分或缩小区间」的策略（仍以工具返回为准）。

## `calculate_energy_intensity_preyear`

- 用途：指定 **自然年** 的单位面积能耗（各类型除以建筑面积 `sqm`）。
- 若建筑不存在或无面积，以服务返回的 `message` / `error` 为准。

## `anomaly_detect`

- **`meter_type`**：与库内一致，如 `electricity`、`chilledwater`、`hotwater`、`water`。
- **`year` / `month`**：整数；用于锁定一个月做序列异常检测。


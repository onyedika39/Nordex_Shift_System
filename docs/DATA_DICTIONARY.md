# Data Dictionary

The main dataset is stored in `ShiftData.db` inside the `ShiftPerformance` table.

## Table: ShiftPerformance

| Column | Description |
| --- | --- |
| `shift_id` | Unique identifier for a production shift. |
| `shift_name` | Name/category of the shift, such as morning, afternoon, or night. |
| `start_time` | Shift start time. |
| `end_time` | Shift end time. |
| `supervisor_id` | Identifier for the shift supervisor. |
| `production_id` | Unique identifier for a production record. |
| `date` | Production date. |
| `units_produced` | Number of units produced during the shift. |
| `defect_count` | Number of defective units recorded. |
| `cycle_time_avg` | Average cycle time for production. |
| `shift_efficiency_score` | Target performance score used for machine learning. |
| `operator_id` | Unique identifier for the operator. |
| `operator_name` | Operator name. |
| `experience_level` | Operator experience level. |
| `skill_category` | Operator skill category. |
| `machine_id` | Unique identifier for the machine. |
| `runtime_hours` | Machine runtime during the shift. |
| `downtime_minutes` | Machine downtime during the shift. |
| `maintenance_flag` | Indicates whether maintenance occurred. |
| `machine_status` | Operational status of the machine. |
| `maintenance_id` | Unique identifier for a maintenance record. |
| `issue_type` | Type of maintenance issue recorded. |
| `maintenance_downtime` | Downtime associated with maintenance. |
| `resolved_by` | Person or team that resolved the maintenance issue. |
| `qc_id` | Quality control identifier. |
| `defect_type` | Type/category of defect. |
| `severity` | Severity level of the quality issue. |
| `inspection_result` | Result of quality inspection. |
| `temperature` | Environmental temperature reading. |
| `humidity` | Environmental humidity reading. |
| `timestamp` | Timestamp for the recorded event. |

## Target Variable

`shift_efficiency_score` is the main prediction target for the regression models.

## Engineered Features

| Feature | Description |
| --- | --- |
| `shift_duration` | Total duration of the shift in hours. |
| `defect_rate` | Defects divided by units produced. |
| `downtime_ratio` | Downtime minutes divided by total shift minutes. |
| `day_of_week` | Day of week extracted from the production date. |
| `hour_of_day` | Hour extracted from the shift start time. |

## Missing Data Notes

The notebook identifies missing values in maintenance-related and environmental columns.

Handling approach:

- Missing `temperature` and `humidity` values are forward-filled and then filled with the column mean.
- Missing maintenance issue fields are filled with business-readable labels such as `No Issue` and `No Maintenance`.
- Missing `maintenance_downtime` values are filled with `0`.

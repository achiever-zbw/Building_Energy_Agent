import io
import pandas as pd


class ReportService:
    """报告生成服务：把查询结果 dict 序列化为可下载的字节流。"""

    @staticmethod
    def _build_dataframe(messages: dict) -> pd.DataFrame:
        """
        将查询返回结构转换为可导出的 DataFrame。

        特殊处理：
        - 预测接口返回 {building_id, meter_type, target_date, forecast:[...]}
          导出时展开为“每个时间点一行”。
        """
        if not isinstance(messages, dict):
            raise ValueError("导出数据格式错误：data 必须是对象")

        forecast = messages.get("forecast")
        if isinstance(forecast, list):
            # 预测结果：按时间点展开行
            base = {
                "building_id": messages.get("building_id"),
                "meter_type": messages.get("meter_type") or "-",
                "target_date": messages.get("target_date"),
            }
            rows = []
            for item in forecast:
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        **base,
                        "timestamp": item.get("timestamp"),
                        "predicted_energy": item.get("predicted_energy"),
                    }
                )
            # 即便 forecast 为空也保留列头
            if not rows:
                return pd.DataFrame(
                    columns=[
                        "building_id",
                        "meter_type",
                        "target_date",
                        "timestamp",
                        "predicted_energy",
                    ]
                )
            return pd.DataFrame(rows)

        # 默认：单行汇总
        return pd.DataFrame([messages])

    @staticmethod
    async def report_file(
        messages: dict,
        file_type: str = "csv",
    ) -> bytes:
        """
        将单行汇总 dict 导出为字节（供 FastAPI Response 使用）。

        - csv: UTF-8 BOM，便于 Excel 打开
        - excel: .xlsx（需依赖 openpyxl）
        - json: orient=records 的单元素数组 JSON
        """
        try:
            df = ReportService._build_dataframe(messages)

            if file_type == "csv":
                buf = io.StringIO()
                df.to_csv(buf, index=False)
                return buf.getvalue().encode("utf-8-sig")

            if file_type == "excel":
                buf = io.BytesIO()
                df.to_excel(buf, index=False, engine="openpyxl")
                return buf.getvalue()

            if file_type == "json":
                s = df.to_json(orient="records", force_ascii=False)
                return s.encode("utf-8")

            raise ValueError(f"不支持的文件格式: {file_type}")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"导出文件失败: {e}") from e
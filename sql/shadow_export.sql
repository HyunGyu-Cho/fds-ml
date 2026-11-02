-- Ⓚ 만약 비교용 Shadow 기록 내보내기 (읽기 전용 SELECT, 돈워리 로컬 DB donworry)
-- DataGrip에서 실행 → 결과 우클릭 Export Data → CSV(머리글 포함) → fds-ml/data/shadow/shadow_export.csv
SELECT s.payment_id,
       s.model_version,
       s.status,
       s.rule_result,
       s.rule_risk_score,
       s.fraud_probability,
       s.threshold,
       s.ml_risk_level,
       s.features,
       e.ip_country_code,
       (SELECT GROUP_CONCAT(h.rule_code ORDER BY h.rule_code SEPARATOR ' ')
        FROM fds_rule_hits h
        WHERE h.evaluation_id = s.evaluation_id
          AND h.outcome = 'MATCH') AS matched_rules,
       d.decision                    AS staff_decision
FROM ml_shadow_scores s
         JOIN fds_evaluations e ON e.evaluation_id = s.evaluation_id
         LEFT JOIN case_payments cp ON cp.payment_id = s.payment_id
         LEFT JOIN payment_decisions d ON d.case_payment_id = cp.case_payment_id
ORDER BY s.payment_id, s.model_version;

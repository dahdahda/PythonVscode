<!--
SKILL.md — 可复用技能模板
遵循 `agent-customization` 指南：清晰、可执行、以步骤为中心，包含决策点与验收标准。
-->

# Skill: Generalize Conversation Workflow

## Summary
将对话中正在执行的多步工作流抽象为一个可复用的技能（SKILL），用于在工作区或个人范围内复用、自动化和校验该流程。

## Scope
- 默认：工作区作用域（workspace-scoped）。
- 可选：个人作用域（personal），在创建时明确说明。

## When To Use
- 会话中出现了可被复用的多步方法、检查清单或调试/实施流程。
- 需要将经验固化为可重复执行的步骤或模板。

## Intended Outcome
生成一个包含：步骤清单、决策逻辑、质量准则、示例提示、以及用于迭代的澄清问题的 SKILL 文件。

## Step-by-step Process
1. 提取对话里正在执行的具体步骤（按顺序）。
2. 为每一步定义：输入、预期产出、完成标准（验收条件）。
3. 标记决策点：列出条件、分支与对应后续步骤。
4. 汇总质量准则：何时认为完成、常见错误与回退策略。
5. 编写示例提示：展示如何用本技能触发自动化或复用流程。
6. 保存为 `SKILL.md` 并放在工作区约定位置（如仓库根目录的 `.vscode/skills/` 或仓库根）。

## Decision Points & Branching
- 对每个决策点，包含：判断条件（布尔或阈值）、首选分支、降级或回退步骤。
- 使用简短的 if/else 模式描述，例如：
  - 如果 X 成立 → 执行步骤 A
  - 否则 → 执行步骤 B（并记录原因）

## Quality Criteria (验收标准)
- 步骤按顺序被完整执行；每一步都有明确的输入与输出。
- 关键断言或检查点通过（列出要运行的命令或测试）。
- 结果被记录于工作区（日志、PR 描述或生成的 artifact）。

## Example Prompts
- "把我们刚才的调试流程抽成技能并生成检查清单。"
- "为 `部署` 相关对话生成 SKILL，包含健康检查与回滚步骤。"

## Ambiguities To Clarify (建议在草案后询问)
- 该技能应为工作区作用域还是个人作用域？
- 是否需要包含自动化脚本或仅文本步骤？
- 期望的文件路径与命名约定是什么？

## Iteration & Finalization
1. 识别最模糊或风险最高的步骤并标注为“需确认”。
2. 向对话参与者提出针对性问题以消除模糊点。
3. 更新 SKILL.md 并记录变更历史（简短变更说明）。

## Related Customizations (建议)
- 为常见任务创建一组 SKILL：`testing`, `deploy`, `code-review`, `debugging`。
- 将 SKILL 与 CI/CD job 模板或 PR 模板关联，便于在仓库内直接复用。

---
Generated-by: agent-customization guide

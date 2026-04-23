import ast


class SafeEvalError(Exception):
    pass


class SafeEvaluator:
    @staticmethod
    def eval_expr(expression, variables=None):
        try:
            tree = ast.parse(expression, mode="eval")
            compiled = compile(tree, "<safe_eval>", "eval")
            return eval(compiled, {"__builtins__": {}}, variables or {})
        except Exception as exc:
            raise SafeEvalError(str(exc)) from exc

//! Deterministic scoring helpers shared by benchmarks.

use regex::Regex;
use std::sync::OnceLock;

fn re(cell: &'static OnceLock<Regex>, pat: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pat).expect("valid regex"))
}

/// A number with optional thousands separators. "1,234" is one number but
/// "1,2" is two, so lists are not glued together.
const NUM: &str = r"-?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?";

fn parse_num(s: &str) -> Option<f64> {
    s.replace(',', "").parse::<f64>().ok()
}

/// Extract the final numeric answer from free text.
///
/// Priority: `\boxed{}`, GSM8K `####` marker, explicit answer phrases, a last
/// sentence that mentions an answer, and finally the last number in the text.
pub fn extract_numerical_answer(text: &str) -> Option<f64> {
    static HASHES: OnceLock<Regex> = OnceLock::new();
    static PHRASES: OnceLock<Vec<Regex>> = OnceLock::new();
    static ANY: OnceLock<Regex> = OnceLock::new();

    if let Some(inner) = extract_boxed(text) {
        let cleaned = inner
            .replace("\\$", "")
            .replace("\\%", "")
            .replace(['$', '%'], "");
        if let Some(v) = parse_num(cleaned.trim()) {
            return Some(v);
        }
    }
    let hashes = re(&HASHES, &format!(r"####\s*({NUM})"));
    if let Some(c) = hashes.captures(text) {
        if let Some(v) = parse_num(&c[1]) {
            return Some(v);
        }
    }

    let phrases = PHRASES.get_or_init(|| {
        [
            format!(r"answer is[:\s]+\$?({NUM})"),
            format!(r"answer:[:\s]*\$?({NUM})"),
            format!(r"= ({NUM})\s*$"),
            format!(r"total[:\s]+\$?({NUM})"),
            format!(r"result[:\s]+\$?({NUM})"),
        ]
        .iter()
        .map(|p| Regex::new(p).expect("valid regex"))
        .collect()
    });
    let lower = text.to_lowercase();
    for p in phrases {
        if let Some(c) = p.captures_iter(&lower).last() {
            if let Some(v) = parse_num(&c[1]) {
                return Some(v);
            }
        }
    }

    let any = re(&ANY, NUM);
    let sentences: Vec<&str> = text.trim().split('.').collect();
    if let Some(last) = sentences.last() {
        let l = last.to_lowercase();
        if ["answer", "total", "result", "therefore"]
            .iter()
            .any(|w| l.contains(w))
        {
            if let Some(m) = any.find_iter(last).last() {
                if let Some(v) = parse_num(m.as_str()) {
                    return Some(v);
                }
            }
        }
    }

    any.find_iter(text)
        .last()
        .and_then(|m| parse_num(m.as_str()))
}

/// Contents of the last `\boxed{...}` (or `\fbox{...}`) with balanced braces.
pub fn extract_boxed(text: &str) -> Option<String> {
    let start = ["\\boxed{", "\\fbox{"]
        .iter()
        .filter_map(|p| text.rfind(p).map(|i| (i, p.len())))
        .max_by_key(|(i, _)| *i)?;
    let begin = start.0 + start.1;
    let mut depth = 1usize;
    for (off, ch) in text[begin..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[begin..begin + off].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// Numeric equality with a small tolerance. NaN or missing values never match.
pub fn is_exact_match(predicted: Option<f64>, truth: Option<f64>) -> bool {
    match (predicted, truth) {
        (Some(p), Some(t)) if p.is_finite() && t.is_finite() => (p - t).abs() < 1e-6,
        _ => false,
    }
}

/// Unbiased pass@k estimator from n samples of which c are correct.
pub fn pass_at_k(n: usize, c: usize, k: usize) -> f64 {
    if k == 0 || n == 0 {
        return 0.0;
    }
    let c = c.min(n);
    if n - c < k {
        return 1.0;
    }
    // 1 - prod_{i = n-c+1}^{n} (1 - k / i), stable for large n.
    let mut prod = 1.0;
    for i in (n - c + 1)..=n {
        prod *= 1.0 - k as f64 / i as f64;
    }
    1.0 - prod
}

/// Wilson score interval for a proportion at 95% confidence.
pub fn wilson_ci(successes: f64, n: usize) -> Option<(f64, f64)> {
    if n == 0 {
        return None;
    }
    let z = 1.959_963_984_540_054_f64;
    let n = n as f64;
    let p = successes / n;
    let denom = 1.0 + z * z / n;
    let centre = p + z * z / (2.0 * n);
    let spread = z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).sqrt();
    Some((
        ((centre - spread) / denom).max(0.0),
        ((centre + spread) / denom).min(1.0),
    ))
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

/// Mean and population standard deviation, matching the original suite.
pub fn mean_std(values: &[f64]) -> Option<Stats> {
    if values.is_empty() {
        return None;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let std = if values.len() > 1 {
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt()
    } else {
        0.0
    };
    Some(Stats {
        mean,
        std,
        min: values.iter().cloned().fold(f64::INFINITY, f64::min),
        max: values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    })
}

/// Normalise a LaTeX or plain math answer for string comparison.
pub fn normalize_math(s: &str) -> String {
    let mut t = s.trim().to_string();
    for (from, to) in [
        ("\\left", ""),
        ("\\right", ""),
        ("\\!", ""),
        ("\\,", ""),
        ("\\;", ""),
        ("\\ ", ""),
        ("\\dfrac", "\\frac"),
        ("\\tfrac", "\\frac"),
        ("^\\circ", ""),
        ("^{\\circ}", ""),
        ("\\%", ""),
        ("\\$", ""),
        ("$", ""),
        ("%", ""),
    ] {
        t = t.replace(from, to);
    }
    // Unwrap \text{...} and \mathrm{...}.
    static TEXT: OnceLock<Regex> = OnceLock::new();
    let text_re = re(&TEXT, r"\\(?:text|mathrm|textbf)\{([^{}]*)\}");
    t = text_re.replace_all(&t, "$1").to_string();
    t.retain(|c| !c.is_whitespace());
    t.trim_end_matches('.').to_string()
}

/// Compare two math answers: normalised string equality or numeric equality.
pub fn math_equal(pred: &str, truth: &str) -> bool {
    let (p, t) = (normalize_math(pred), normalize_math(truth));
    if p == t {
        return true;
    }
    match (math_value(&p), math_value(&t)) {
        (Some(a), Some(b)) => (a - b).abs() < 1e-6 * b.abs().max(1.0),
        _ => false,
    }
}

/// Numeric value of a normalised answer: plain numbers (thousands separators
/// only in the 1,234 form), `\frac{a}{b}`, `\frac12` and `a/b`.
fn math_value(s: &str) -> Option<f64> {
    static NUMBER: OnceLock<Regex> = OnceLock::new();
    static FRAC: OnceLock<Regex> = OnceLock::new();
    static SLASH: OnceLock<Regex> = OnceLock::new();
    let number = re(&NUMBER, &format!("^{NUM}$"));
    let num = |x: &str| number.is_match(x).then(|| parse_num(x)).flatten();
    if let Some(v) = num(s) {
        return Some(v);
    }
    let frac = re(
        &FRAC,
        r"^(-?)\\frac(?:\{([^{}]+)\}|([0-9]))(?:\{([^{}]+)\}|([0-9]))$",
    );
    if let Some(c) = frac.captures(s) {
        let a = num(c.get(2).or(c.get(3))?.as_str())?;
        let b = num(c.get(4).or(c.get(5))?.as_str())?;
        if b == 0.0 {
            return None;
        }
        let v = a / b;
        return Some(if &c[1] == "-" { -v } else { v });
    }
    let slash = re(&SLASH, r"^(-?[0-9.,]+)/([0-9.,]+)$");
    if let Some(c) = slash.captures(s) {
        let (a, b) = (num(&c[1])?, num(&c[2])?);
        return (b != 0.0).then(|| a / b);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gsm8k_marker_wins() {
        assert_eq!(
            extract_numerical_answer("so 3*4=12\n#### 1,234"),
            Some(1234.0)
        );
    }

    #[test]
    fn boxed_answer() {
        assert_eq!(
            extract_numerical_answer("thus \\boxed{42} is it. Also 7"),
            Some(42.0)
        );
        assert_eq!(extract_numerical_answer("\\boxed{\\$1,500}"), Some(1500.0));
    }

    #[test]
    fn answer_phrases() {
        assert_eq!(extract_numerical_answer("The answer is 18."), Some(18.0));
        assert_eq!(extract_numerical_answer("Answer: 3.5"), Some(3.5));
        assert_eq!(extract_numerical_answer("we get 5 + 6 = 11"), Some(11.0));
    }

    #[test]
    fn last_number_fallback_and_none() {
        assert_eq!(
            extract_numerical_answer("I have 2 apples and 9 pears"),
            Some(9.0)
        );
        assert_eq!(extract_numerical_answer("no digits here"), None);
    }

    #[test]
    fn negative_numbers() {
        assert_eq!(extract_numerical_answer("The answer is -4"), Some(-4.0));
    }

    #[test]
    fn list_commas_are_not_thousands() {
        assert_eq!(extract_numerical_answer("values 5,6"), Some(6.0));
        assert_eq!(
            extract_numerical_answer("so 1,234,567 total"),
            Some(1234567.0)
        );
    }

    #[test]
    fn pass_at_k_clamps_c() {
        assert_eq!(pass_at_k(3, 9, 1), 1.0);
    }

    #[test]
    fn exact_match_tolerance() {
        assert!(is_exact_match(Some(2.0), Some(2.0000001)));
        assert!(!is_exact_match(Some(2.0), Some(2.1)));
        assert!(!is_exact_match(None, Some(2.0)));
        assert!(!is_exact_match(Some(f64::NAN), Some(f64::NAN)));
    }

    #[test]
    fn pass_at_k_known_values() {
        assert_eq!(pass_at_k(10, 0, 1), 0.0);
        assert_eq!(pass_at_k(10, 10, 1), 1.0);
        assert!((pass_at_k(10, 5, 1) - 0.5).abs() < 1e-12);
        // n=5, c=1, k=2: 1 - C(4,2)/C(5,2) = 1 - 6/10
        assert!((pass_at_k(5, 1, 2) - 0.4).abs() < 1e-12);
        assert_eq!(pass_at_k(5, 4, 2), 1.0);
    }

    #[test]
    fn wilson_bounds() {
        let (lo, hi) = wilson_ci(50.0, 100).unwrap();
        assert!(lo < 0.5 && hi > 0.5);
        assert!((lo - 0.4038).abs() < 1e-3, "lo={lo}");
        assert!((hi - 0.5962).abs() < 1e-3, "hi={hi}");
        let (lo, hi) = wilson_ci(0.0, 20).unwrap();
        assert_eq!(lo, 0.0);
        assert!(hi > 0.0);
        assert!(wilson_ci(0.0, 0).is_none());
    }

    #[test]
    fn stats() {
        let s = mean_std(&[1.0, 3.0]).unwrap();
        assert_eq!(s.mean, 2.0);
        assert_eq!(s.std, 1.0);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 3.0);
        assert!(mean_std(&[]).is_none());
    }

    #[test]
    fn boxed_nested_braces() {
        assert_eq!(
            extract_boxed("x \\boxed{\\frac{1}{2}} y").as_deref(),
            Some("\\frac{1}{2}")
        );
        assert!(extract_boxed("no box").is_none());
    }

    #[test]
    fn math_equality() {
        assert!(math_equal("\\dfrac{1}{2}", "\\frac{1}{2}"));
        assert!(math_equal("0.50", "0.5"));
        assert!(math_equal("45^\\circ", "45"));
        assert!(math_equal("\\text{(A)}", "(A)"));
        assert!(!math_equal("3", "4"));
        assert!(math_equal("\\frac12", "\\frac{1}{2}"));
        assert!(math_equal("1/2", "\\frac{1}{2}"));
        assert!(math_equal("0.5", "\\frac{1}{2}"));
        assert!(math_equal("-\\frac{3}{4}", "-0.75"));
        assert!(math_equal("1,000", "1000"));
        // a list is not the number with its commas removed
        assert!(!math_equal("1,2", "12"));
        assert!(!math_equal("\\frac{1}{0}", "0"));
    }
}

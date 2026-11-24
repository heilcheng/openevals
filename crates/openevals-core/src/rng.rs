//! Small deterministic RNG so shuffles are stable across platforms and
//! dependency versions.

pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}

/// FNV-1a hash of a string mixed with a base seed, for per-item seeds.
pub fn seed_for(key: &str, base: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ base;
    for b in key.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_is_deterministic() {
        let mut a: Vec<u32> = (0..10).collect();
        let mut b = a.clone();
        SplitMix64::new(7).shuffle(&mut a);
        SplitMix64::new(7).shuffle(&mut b);
        assert_eq!(a, b);
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn seed_depends_on_key() {
        assert_ne!(seed_for("a", 1), seed_for("b", 1));
        assert_eq!(seed_for("a", 1), seed_for("a", 1));
    }
}

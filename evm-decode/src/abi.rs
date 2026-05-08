use alloy_json_abi::JsonAbi;
use anyhow::{Context, Result};

/// Parsed event info extracted from a JSON ABI.
#[derive(Debug, Clone)]
pub struct EvmAbiEvent {
    /// Event name (e.g. "Swap").
    pub name: String,
    /// Event name in snake_case (e.g. "swap").
    pub name_snake_case: String,
    /// Human-readable signature with outer param names but unnamed tuple components
    /// (e.g. `"Swap(address indexed sender, (int256,int256) data)"`).
    /// Can be passed directly to [`crate::decode_events`] and [`crate::signature_to_topic0`].
    pub signature: String,
    /// Canonical selector signature without names
    /// (e.g. "Swap(address,address,int256,int256,uint160,uint128,int24)").
    pub selector_signature: String,
    /// topic0 as 0x-prefixed hex string.
    pub topic0: String,
    /// Full JSON ABI fragment for this event, preserving all component names.
    pub abi_json: String,
}

/// Parsed function info extracted from a JSON ABI.
#[derive(Debug, Clone)]
pub struct EvmAbiFunction {
    /// Function name (e.g. "swap").
    pub name: String,
    /// Function name in snake_case (e.g. "swap").
    pub name_snake_case: String,
    /// Human-readable signature with names
    /// (e.g. "swap(address recipient, bool zeroForOne, int256 amountSpecified, ...)").
    pub signature: String,
    /// Canonical selector signature without names
    /// (e.g. "swap(address,bool,int256,uint160,bytes)").
    pub selector_signature: String,
    /// 4-byte selector as 0x-prefixed hex string.
    pub selector: String,
    /// Full JSON ABI fragment for this function, preserving all parameter names.
    pub abi_json: String,
}

/// Converts a camelCase or PascalCase name to snake_case.
fn to_snake_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for (i, ch) in name.char_indices() {
        if ch.is_uppercase() && i != 0 {
            out.push('_');
        }
        out.extend(ch.to_lowercase());
    }
    out
}

/// Converts a param's type and components to a string without inner names,
/// e.g. `tuple(int256,uint256)` → `(int256,uint256)`, `tuple[]` → `(int256,uint256)[]`.
fn param_type_str(ty: &str, components: &[alloy_json_abi::Param]) -> String {
    if !components.is_empty() && ty.starts_with("tuple") {
        let suffix = &ty["tuple".len()..]; // "" | "[]" | "[3]" | …
        let inner = components
            .iter()
            .map(|c| param_type_str(&c.ty, &c.components))
            .collect::<Vec<_>>()
            .join(",");
        format!("tuple({inner}){suffix}")
    } else {
        ty.to_string()
    }
}

/// Builds a human-readable event signature: outer param names and `indexed` flags kept,
/// inner tuple component names stripped.
///
/// Example: `Swap(address indexed sender, tuple(int256,int256) delta)`
fn event_decode_signature(event: &alloy_json_abi::Event) -> String {
    let params = event
        .inputs
        .iter()
        .map(|input| {
            let ty = param_type_str(&input.ty, &input.components);
            match (input.indexed, input.name.is_empty()) {
                (true, false) => format!("{ty} indexed {}", input.name),
                (true, true) => format!("{ty} indexed"),
                (false, false) => format!("{ty} {}", input.name),
                (false, true) => ty,
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{}({params})", event.name)
}

/// Builds a human-readable function signature: outer param names kept, inner tuple names stripped.
///
/// Example: `swap(address recipient, tuple(int256,uint256) params)(bool success)`
fn func_decode_signature(func: &alloy_json_abi::Function) -> String {
    let fmt_params = |params: &[alloy_json_abi::Param]| {
        params
            .iter()
            .map(|p| {
                let ty = param_type_str(&p.ty, &p.components);
                if p.name.is_empty() {
                    ty
                } else {
                    format!("{ty} {}", p.name)
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!("{}({})({})", func.name, fmt_params(&func.inputs), fmt_params(&func.outputs))
}

/// Parse a JSON ABI string and extract all events.
pub fn abi_events(json_str: &str) -> Result<Vec<EvmAbiEvent>> {
    let abi: JsonAbi = serde_json::from_str(json_str).context("parse JSON ABI")?;
    let mut events = Vec::new();
    for event in abi.events() {
        let selector = event.selector();
        events.push(EvmAbiEvent {
            name_snake_case: to_snake_case(&event.name),
            name: event.name.clone(),
            signature: event_decode_signature(event),
            selector_signature: event.signature(),
            topic0: format!("0x{}", faster_hex::hex_string(selector.as_slice())),
            abi_json: serde_json::to_string(event)
                .expect("alloy_json_abi::Event serialization is infallible"),
        });
    }
    Ok(events)
}

/// Parse a JSON ABI string and extract all functions.
pub fn abi_functions(json_str: &str) -> Result<Vec<EvmAbiFunction>> {
    let abi: JsonAbi = serde_json::from_str(json_str).context("parse JSON ABI")?;
    let mut functions = Vec::new();
    for func in abi.functions() {
        let selector = func.selector();
        functions.push(EvmAbiFunction {
            name_snake_case: to_snake_case(&func.name),
            name: func.name.clone(),
            signature: func_decode_signature(func),
            selector_signature: func.signature(),
            selector: format!("0x{}", faster_hex::hex_string(selector.as_slice())),
            abi_json: serde_json::to_string(func)
                .expect("alloy_json_abi::Function serialization is infallible"),
        });
    }
    Ok(functions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature_to_topic0;

    #[test]
    fn test_abi_events() {
        let abi_json = r#"[
            {
                "anonymous": false,
                "inputs": [
                    {"indexed": true, "internalType": "address", "name": "sender", "type": "address"},
                    {"indexed": true, "internalType": "address", "name": "recipient", "type": "address"},
                    {"indexed": false, "internalType": "int256", "name": "amount0", "type": "int256"},
                    {"indexed": false, "internalType": "int256", "name": "amount1", "type": "int256"},
                    {"indexed": false, "internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
                    {"indexed": false, "internalType": "uint128", "name": "liquidity", "type": "uint128"},
                    {"indexed": false, "internalType": "int24", "name": "tick", "type": "int24"}
                ],
                "name": "Swap",
                "type": "event"
            },
            {
                "inputs": [{"internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"}],
                "name": "initialize",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]"#;

        let events = abi_events(abi_json);
        assert!(events.is_ok());
        let events = events.unwrap_or_default();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "Swap");
        assert_eq!(
            events[0].signature,
            "Swap(address indexed sender, address indexed recipient, int256 amount0, int256 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick)"
        );
        assert_eq!(
            events[0].selector_signature,
            "Swap(address,address,int256,int256,uint160,uint128,int24)"
        );
        assert!(events[0].topic0.starts_with("0x"));
        // Verify the signature works with decode (parse round-trip)
        let topic0_from_sig = signature_to_topic0(&events[0].signature);
        assert!(topic0_from_sig.is_ok());
    }

    #[test]
    fn test_abi_functions() {
        let abi_json = r#"[
            {
                "inputs": [
                    {"internalType": "address", "name": "recipient", "type": "address"},
                    {"internalType": "bool", "name": "zeroForOne", "type": "bool"},
                    {"internalType": "int256", "name": "amountSpecified", "type": "int256"},
                    {"internalType": "uint160", "name": "sqrtPriceLimitX96", "type": "uint160"},
                    {"internalType": "bytes", "name": "data", "type": "bytes"}
                ],
                "name": "swap",
                "outputs": [
                    {"internalType": "int256", "name": "amount0", "type": "int256"},
                    {"internalType": "int256", "name": "amount1", "type": "int256"}
                ],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "anonymous": false,
                "inputs": [
                    {"indexed": true, "internalType": "address", "name": "sender", "type": "address"}
                ],
                "name": "Swap",
                "type": "event"
            }
        ]"#;

        let functions = abi_functions(abi_json);
        assert!(functions.is_ok());
        let functions = functions.unwrap_or_default();
        assert_eq!(functions.len(), 1);
        assert_eq!(functions[0].name, "swap");
        assert_eq!(
            functions[0].selector_signature,
            "swap(address,bool,int256,uint160,bytes)"
        );
        assert!(functions[0].selector.starts_with("0x"));
        assert_eq!(functions[0].selector.len(), 10); // "0x" + 8 hex chars
    }

    #[test]
    fn test_abi_events_empty_abi() {
        let events = abi_events("[]");
        assert!(events.is_ok());
        assert!(events.unwrap_or_default().is_empty());
    }
}

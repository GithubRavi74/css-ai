def show_dashboard(final_detections, violators, total_frames):
    st.write("---")
    st.subheader("📊 Safety Analytics Summary")
    
    if len(final_detections) > 0 and total_frames > 0:
        raw_counts = Counter(final_detections)
        avg_counts = {item: max(1, round(count / total_frames)) for item, count in raw_counts.items()}
        has_violation = any(v in avg_counts for v in violators)
        
        if has_violation:
            st.error("⚠️ **Safety Compliance Alert:** The model detected ongoing missing or inadequate PPE on site personnel.")
        else:
            st.success("✅ **Compliance Passed:** Site personnel are consistently equipped with standard safety gear.")
            
        col1, col2, col3 = st.columns(3)
        with col1:
            total_avg_objects = sum(avg_counts.values())
            st.metric(label="Avg Objects Spotted/Frame", value=total_avg_objects)
        with col2:
            st.metric(label="Estimated Workers on Site", value=avg_counts.get("Person", 0))
        with col3:
            total_violations = sum(avg_counts[v] for v in violators if v in avg_counts)
            safety_score = max(0, 100 - (total_violations * 25)) 
            st.metric(label="Site Compliance Score", value=f"{safety_score}%")

        st.markdown("### 📋 Detailed Inspection Ledger (Averaged Stream Data)")
        
        # 1. Initialize table header structure and CSS styles
        table_html = """
        <style>
            .professional-table {
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                font-size: 15px;
                font-family: sans-serif;
            }
            .professional-table thead tr {
                background-color: #1E293B !important;
                color: #FFFFFF !important;
                font-weight: bold;
            }
            .professional-table th, .professional-table td {
                padding: 12px 15px;
                border: 1px solid #E2E8F0;
                text-align: center !important;
            }
            .professional-table tbody tr {
                border-bottom: 1px solid #E2E8F0;
            }
            .bold-cell {
                font-weight: bold !important;
            }
        </style>
        <table class="professional-table">
            <thead>
                <tr>
                    <th>Identified Object</th>
                    <th>Avg Quantity On-Screen</th>
                    <th>Operational Status</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # 2. Build out rows dynamically as fully structured text chunks
        for item, count in avg_counts.items():
            if item in violators:
                status = "🔴 Violation / Risk Factor"
            elif item in ["Hardhat", "Mask", "Safety Vest"]:
                status = "🟢 Compliant Protection"
            else:
                status = "🔵 Registered Asset"
                
            row_html = f"""
                <tr>
                    <td class="bold-cell">{item}</td>
                    <td class="bold-cell">{count}</td>
                    <td>{status}</td>
                </tr>
            """
            table_html += row_html
            
        # 3. Securely close out the document nodes
        table_html += """
            </tbody>
        </table>
        """
        
        # 4. Pass the singular, comprehensive string directly to the layout parser
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("Scan clear. No personnel or assets were registered in this file.")
